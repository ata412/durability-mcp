"""E19: pre-start, real-only next eligible session magnitude prediction."""
from __future__ import annotations

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path
from durability.evaluate import assert_group_split_integrity, regression_metrics, shuffle_targets_within_athlete
from durability.level_change import markdown_table
from durability.models.exploratory import make_exploratory_model
from durability.models.tier1_gbdt import make_gbdt_model
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER=logging.getLogger(__name__)
PROVENANCE='PILOT MEASURED EXPLORATORY ON REAL NEXT ELIGIBLE RUNS'


def build_previous_features(frame, config):
    """Only completed earlier eligible sessions; simultaneous starts excluded together."""
    settings=config['next_run_prediction']; outputs=settings['prior_outcomes']
    ordered=frame.copy()
    ordered['start_time']=pd.to_datetime(ordered.start_time,utc=True)
    ordered['end_time']=ordered.start_time+pd.to_timedelta(ordered.duration_s,unit='s')
    ordered=ordered.sort_values(['athlete_id','start_time','activity_id'])
    rows=[]
    for athlete, group in ordered.groupby('athlete_id',sort=True):
        history=[]
        for time, tied in group.groupby('start_time',sort=True):
            prior=[r for r in history if r['end_time']<time]
            for current in tied.to_dict('records'):
                row=dict(activity_id=current['activity_id'],athlete_id=athlete,
                    prior_session_count=len(prior),
                    prior_max_start_time=max((r['start_time'] for r in prior),default=pd.NaT),
                    prior_max_end_time=max((r['end_time'] for r in prior),default=pd.NaT),
                    previous_activity_id=prior[-1]['activity_id'] if prior else None)
                for col in outputs:
                    name=col.removeprefix('target_'); values=np.array([r[col] for r in prior],dtype=float)
                    finite=values[np.isfinite(values)]
                    row[f'previous_{name}']=values[-1] if len(values) else np.nan
                    row[f'prior_{name}_mean']=finite.mean() if len(finite) else np.nan
                    row[f'prior_{name}_sd']=finite.std(ddof=1) if len(finite)>1 else np.nan
                    recent=values[-settings['recent_sessions']:]
                    row[f'recent_{name}_mean']=np.nanmean(recent) if np.isfinite(recent).any() else np.nan
                row['previous_duration_min']=prior[-1]['duration_s']/60 if prior else np.nan
                row['previous_gap_speed_cv']=prior[-1]['gap_speed_cv'] if prior else np.nan
                row['days_since_previous']=(time-prior[-1]['end_time']).total_seconds()/86400 if prior else np.nan
                rows.append(row)
            history.extend(tied.to_dict('records'))
    result=pd.DataFrame(rows)
    return frame[['activity_id','athlete_id']].merge(result,on=['activity_id','athlete_id'],validate='one_to_one')


def predictor_columns(manifest_columns, prior, config):
    prior_numeric=[c for c in prior if c not in
        {'activity_id','athlete_id','prior_max_start_time','prior_max_end_time','previous_activity_id'}]
    previous=[c for c in prior_numeric if c.startswith('previous_') or c=='days_since_previous']
    # Explicit prefixes / exact parameter allowlist, never take arbitrary non-early columns.
    settings=config['experiments']['feature_blocks']
    prefixes=tuple(settings['training_load_prefixes']+settings['prior_durability_prefixes'])
    parameters=set(settings['athlete_parameter_columns'])
    history=[c for c in manifest_columns if c.startswith(prefixes) or c in parameters or c=='history_activity_count']
    arms={'previous_only':previous,'history_combined':list(dict.fromkeys(history+prior_numeric))}
    assert all(not c.startswith(('early_','target_')) for cols in arms.values() for c in cols)
    return arms


def completed_history_mask(frame, activities):
    """Conservatively exclude targets whose cached start-based history includes unfinished runs."""
    activities=activities[activities.accepted.eq(True)].copy()
    activities['start_time']=pd.to_datetime(activities.start_time,utc=True)
    activities['end_time']=activities.start_time+pd.to_timedelta(activities.duration_s,unit='s')
    histories={str(a):g.sort_values('start_time') for a,g in activities.groupby('athlete_id')}
    result=[]
    for row in frame.itertuples(index=False):
        g=histories.get(str(row.athlete_id))
        if g is None: result.append(False); continue
        times=g.start_time.to_numpy(); pos=np.searchsorted(times,row.start_time,side='left')
        result.append(bool(pos and g.iloc[:pos].end_time.max()<row.start_time))
    return np.asarray(result)


def make_model(name,config):
    maker=make_gbdt_model if name.startswith('tier1_') else make_exploratory_model
    return maker(name,config,seed=config['seed'])


def summarise(predictions, leakage, config):
    settings=config['next_run_prediction']; summaries=[]; contrasts=[]
    status=leakage.set_index('model').halted.to_dict()
    for cohort in ['all_eligible','strict_quality']:
        data=predictions if cohort=='all_eligible' else predictions[predictions.strict_quality]
        athletes=sorted(data.athlete_id.unique())
        weights=bootstrap_weights(len(athletes),settings['bootstrap_n'],config['seed'])
        errors={}
        for model,part in data.groupby('model'):
            error=abs(part.observed-part.predicted).groupby(part.athlete_id).mean()
            errors[model]=error
            summaries.append(dict(cohort=cohort,model=model,**regression_metrics(part.observed,part.predicted),
                mae_athlete=error.mean(),n_athletes=len(error),
                status='HALTED_SHUFFLE_RULE' if status.get(model,False) else 'ACCEPTED',provenance=PROVENANCE))
        learned=[m for m in status if not status[m]]
        pairs=[(m,b) for m in learned for b in ['population_mean','history_mean']]
        pairs += [(f'{m}_history_combined',f'{m}_previous_only') for m in settings['models']
                  if not status[f'{m}_history_combined'] and not status[f'{m}_previous_only']]
        for challenger,reference in pairs:
            delta=errors[challenger]-errors[reference]; point=delta.mean()
            draws=weighted_bootstrap(delta,athletes,weights)
            contrasts.append(dict(cohort=cohort,challenger=challenger,reference=reference,
                delta_mae_athlete=point,**inference(point,draws,settings['confidence_level']),
                n_athletes=len(athletes),provenance=PROVENANCE))
    comparison=pd.DataFrame(contrasts)
    if not comparison.empty:
        comparison['q_bh']=multipletests(comparison.p_bootstrap,method='fdr_bh')[1]
        comparison['supported_gain']=comparison.ci_high.lt(0)&comparison.q_bh.le(settings['fdr_alpha'])
    else:
        comparison=pd.DataFrame(columns=['cohort','challenger','reference','delta_mae_athlete',
            'ci_low','ci_high','p_bootstrap','q_bh','supported_gain'])
    return pd.DataFrame(summaries),comparison


def run_next_run(config):
    settings=config['next_run_prediction']; paths=settings['outputs']
    # Reuse audited real artifacts; no earlier experiment is recomputed or overwritten.
    frame=pd.read_parquet(repo_path(config,config['quality_cohort_prediction']['outputs']['cohort']))
    frame['start_time']=pd.to_datetime(frame.start_time,utc=True)
    for c in ['activity_id','athlete_id']: frame[c]=frame[c].astype(str)
    assert not frame.duplicated(['activity_id','athlete_id']).any()
    assert frame.gate_pass.all()
    assert (frame.max_history_start_time<frame.start_time).all()
    original_n=len(frame)
    prior=build_previous_features(frame,config)
    manifest=pd.read_csv(repo_path(config,config['outputs']['feature_manifest']))
    arms=predictor_columns(manifest.loc[manifest.role.eq('feature'),'column'].tolist(),prior,config)
    frame=frame.merge(prior,on=['activity_id','athlete_id'],validate='one_to_one')
    activities=pd.read_parquet(repo_path(config,str(repo_path(config,config['data']['interim_dir'])/'activities.parquet')))
    complete=completed_history_mask(frame,activities)
    eligible=frame.prior_session_count.ge(settings['minimum_prior_sessions'])&complete
    frame=frame.loc[eligible].reset_index(drop=True)
    assert (frame.prior_max_start_time<frame.start_time).all()
    assert (frame.prior_max_end_time<frame.start_time).all()
    assert frame.groupby('athlete_id').outer_fold.nunique().eq(1).all()
    # Only persist permitted features, labels and audit metadata, not current target channels.
    legal=list(dict.fromkeys(c for cols in arms.values() for c in cols))
    keep=['activity_id','athlete_id','start_time','outer_fold','strict_quality',settings['target'],
          'previous_activity_id','prior_max_start_time','prior_max_end_time']+legal
    frame=frame[list(dict.fromkeys(keep))].copy()
    shuffled=shuffle_targets_within_athlete(frame,[settings['target']],seed=config['seed'])
    pd.testing.assert_frame_equal(frame[legal],shuffled[legal])
    predictions=[]; probe=[]; fold_counts=[]
    for fold in sorted(frame.outer_fold.unique()):
        train=np.flatnonzero(frame.outer_fold.ne(fold)); test=np.flatnonzero(frame.outer_fold.eq(fold))
        assert_group_split_integrity(frame,train,test,group_col='athlete_id')
        a,b=frame.iloc[train],frame.iloc[test]; y=a[settings['target']].to_numpy()
        population=y.mean()
        baseline=dict(population_mean=np.full(len(b),population),
            previous_value=b.previous_magnitude.to_numpy(),history_mean=b.prior_magnitude_mean.to_numpy(),
            shrunk_history=(b.prior_session_count*b.prior_magnitude_mean+settings['shrinkage_strength']*population)
                         /(b.prior_session_count+settings['shrinkage_strength']))
        affine=np.linalg.lstsq(np.c_[np.ones(len(a)),a.previous_magnitude],y,rcond=None)[0]
        baseline['calibrated_previous']=np.c_[np.ones(len(b)),b.previous_magnitude]@affine
        estimates=baseline
        fold_counts.append(dict(outer_fold=fold,n_train=len(a),n_test=len(b),
            n_train_athletes=a.athlete_id.nunique(),n_test_athletes=b.athlete_id.nunique(),
            n_strict_test=int(b.strict_quality.sum()),provenance='MEASURED COHORT COUNTS'))
        for model in settings['models']:
            for arm in settings['feature_arms']:
                name=f'{model}_{arm}'; columns=arms[arm]
                LOGGER.info('E19 fold=%s model=%s train=%s test=%s',fold,name,len(a),len(b))
                fit=make_model(model,config); fit.fit(a[columns],y)
                estimates[name]=fit.predict(b[columns])
                negative=make_model(model,config)
                negative.fit(a[columns],shuffled.iloc[train][settings['target']])
                probe.append(pd.DataFrame(dict(model=name,outer_fold=fold,athlete_id=b.athlete_id,
                    observed=shuffled.iloc[test][settings['target']],predicted=negative.predict(b[columns]))))
        for model,values in estimates.items():
            assert np.isfinite(values).all()
            part=b[['activity_id','athlete_id','start_time','outer_fold','strict_quality']].copy()
            part['model']=model; part['observed']=b[settings['target']].to_numpy()
            part['predicted']=np.asarray(values); part['provenance']=PROVENANCE
            predictions.append(part)
    predictions=pd.concat(predictions,ignore_index=True); probe=pd.concat(probe,ignore_index=True)
    leakage=pd.DataFrame([dict(model=m,**regression_metrics(p.observed,p.predicted),
        provenance='MEASURED WITHIN-ATHLETE SHUFFLED TARGET') for m,p in probe.groupby('model')])
    leakage['halted']=leakage.r_squared.gt(config['evaluate']['leakage_r2_max'])
    summary,contrasts=summarise(predictions,leakage,config)
    audit=pd.DataFrame([
        dict(check='input_gate_cohort',n=original_n,violations=0),
        dict(check='excluded_overlap_or_no_completed_history',n=original_n-len(frame),violations=0),
        dict(check='strict_prior_starts_and_ends',n=len(frame),violations=0),
        dict(check='target_input_fields',n=len(legal),violations=0),
        dict(check='athlete_split_overlap',n=frame.outer_fold.nunique(),violations=0)])
    audit['provenance']='MEASURED AUDIT'
    feature_manifest=pd.DataFrame([dict(arm=arm,column=c,role='feature',availability='strictly prior completed activity')
        for arm,columns in arms.items() for c in columns])
    for key,table in [('examples',frame),('predictions',predictions),('manifest',feature_manifest),
                      ('summary',summary),('contrasts',contrasts),('leakage',leakage),
                      ('audit',audit),('folds',pd.DataFrame(fold_counts))]:
        path=repo_path(config,paths[key]); path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.parquet': table.to_parquet(path,index=False)
        else: table.to_csv(path,index=False)
    plot=summary[summary.cohort.eq('all_eligible')].sort_values('mae_athlete')
    fig,ax=plt.subplots(figsize=(11,6))
    ax.barh(plot.model,plot.mae_athlete,color=['#C94C4C' if s!='ACCEPTED' else '#0072B2' for s in plot.status])
    ax.invert_yaxis(); ax.set_xlabel('Equal-athlete magnitude MAE (lower is better; dimensionless)')
    ax.set_title('E19 — pre-start prediction of next gate-eligible run\nFitRec PILOT MEASURED EXPLORATORY; red = halted diagnostic')
    for i,v in enumerate(plot.mae_athlete): ax.text(v,i,f' {v:.5f}',va='center',fontsize=9)
    ax.set_xlim(0,plot.mae_athlete.max()*1.18)
    fig.tight_layout(); fig.savefig(repo_path(config,paths['figure']),dpi=180); plt.close(fig)
    primary=summary[summary.cohort.eq('all_eligible')&summary.model.eq(settings['primary_model'])].iloc[0]
    controls=summary[summary.cohort.eq('all_eligible')].set_index('model')
    reduction=100*(1-primary.mae_athlete/controls.loc['history_mean','mae_athlete'])
    population_reduction=100*(1-primary.mae_athlete/controls.loc['population_mean','mae_athlete'])
    last_reduction=100*(1-primary.mae_athlete/controls.loc['previous_value','mae_athlete'])
    primary_probe=leakage.set_index('model').loc[settings['primary_model'],'r_squared']
    primary_contrast=contrasts[contrasts.cohort.eq('all_eligible')&
        contrasts.challenger.eq(settings['primary_model'])&contrasts.reference.eq('population_mean')]
    report=f'''# E19 — Previous running history predicts the next eligible run

All results: **{PROVENANCE}**. Seed: {config['seed']}. No augmentation.
[Locked protocol](../../../docs/E19_PROTOCOL.md).

## Question and measured cohort

Pre-start regression of **magnitude = late decoupling / early decoupling**.
The global model sees no current target stream, no early reference, actual target duration,
pace, or QC flags. Prior means and full previous-session outcomes are legal once completed.
Prediction time is immediately before the next target starts, not frozen immediately after
the previous activity. Targets are next **gate-eligible** sessions: other runs may intervene.
Gate/QC eligibility is determined retrospectively, so this is conditional evaluation, NOT
an all-run deployable fatigue predictor or a prediction of gate membership.

MEASURED cohort: {len(frame)} target sessions / {frame.athlete_id.nunique()} athletes;
strict E15 subgroup: {int(frame.strict_quality.sum())} sessions /
{frame.loc[frame.strict_quality,'athlete_id'].nunique()} athletes.
Excluded from the original gated cohort: {original_n-len(frame)} for missing completed
eligible history or overlapping accepted activity. Predictor counts:
{ {a:len(c) for a,c in arms.items()} } (MEASURED).

## Primary result

Prespecified {settings['primary_model']}: status **{primary.status}**,
equal-athlete MAE {primary.mae_athlete:.6f}, session R² {primary.r_squared:.6f}.
Versus historical-mean baseline: {reduction:+.2f}% error reduction by point estimate.
This is not classification accuracy. A primary halt is a primary failure, not permission
to substitute the best secondary model. No outer-test tuning or winner selection.

### How much better, and compared with what?

All following numbers are **PILOT MEASURED EXPLORATORY**:

- Primary model vs copying the last eligible magnitude: {last_reduction:.2f}% lower
  equal-athlete MAE (descriptive point estimate, not a prespecified paired superiority test).
- Primary vs population mean: only {population_reduction:.2f}% lower equal-athlete MAE.
  The prespecified paired comparison does not establish benefit unless its supported_gain
  flag is true; see the exact comparison below.
- Historical mean is a weak comparator here; a {reduction:.2f}% reduction against it
  must NOT be presented as that much improvement over a strong simple prediction.
- Shrunk history MAE {controls.loc['shrunk_history','mae_athlete']:.6f} and affine-calibrated
  previous MAE {controls.loc['calibrated_previous','mae_athlete']:.6f} are approximately as
  good as, or slightly better than, primary MAE {primary.mae_athlete:.6f} by point estimate.
- Primary real R² {primary.r_squared:.6f} vs shuffled R² {primary_probe:.6f}. The latter
  preserves athlete structure; passing the 0.10 safeguard is NOT evidence that the real
  model exceeds this null. No session-specific temporal prediction claim is established
  by these two scores. A repeated hierarchical audit is a follow-up, not a completed test.

{markdown_table(primary_contrast)}

## Real held-out results

{markdown_table(summary)}

## Paired equal-athlete contrasts

Negative differences favour challenger. Whole-athlete bootstrap, pointwise confidence
intervals, approximate centered-bootstrap p-values, BH across all broad/strict contrasts.
Halted learned configurations are omitted from these accepted inferential comparisons.

{markdown_table(contrasts)}

## Shuffled-target refit controls

Predictors are frozen before shuffling; prior outcome features are NOT rebuilt from
artificial labels. Within-athlete shuffle retains athlete-level signal. The configured
R² halt threshold is {config['evaluate']['leakage_r2_max']}; flagged scores are diagnostics
only and excluded from accepted claims. Passing is not proof of all leakage absence.

{markdown_table(leakage)}

## Temporal audit and fold sizes

{markdown_table(audit)}

{markdown_table(pd.DataFrame(fold_counts))}

## Interpretation and limits

Magnitude above one indicates an increase in normalized internal cost relative to external
workload, not directly percent speed loss. Historical means can be stronger than naive
copying of the previous session because individual sessions are noisy. Low R² indicates
limited prediction of next-session variability even if MAE improves. Equal-athlete and
session metrics differ: prolific athletes do not dominate the primary metric.

E15–E18 used target early measurements and primarily additive change; their headline
scores are not interchangeable with E19 magnitude or a clean before/after comparison.
The strict-quality subgroup is secondary scoring of the same broad-trained models,
not another strict-trained experiment. Its eligibility also uses the target retrospectively.
No future workout plan is available; actual target duration/intensity is not substituted.
No all-run deployment, causal, laboratory-validity or independent generalisation claim.
Athlete-held-out CV is not a calendar-held-out backtest: training records of other runners
may postdate a test target. Next step: lock the model and validate prospectively on
independent runners, with planned workout descriptors available before starting.

![Measured next-run errors](../../figures/e19_next_run_prediction.png)
'''
    repo_path(config,paths['report']).write_text(report,encoding='utf-8')
    if leakage.halted.any(): LOGGER.warning('E19 halted configurations: %s',leakage.loc[leakage.halted,'model'].tolist())
    return summary
