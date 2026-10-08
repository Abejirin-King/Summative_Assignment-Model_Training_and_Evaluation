import pandas as pd
import numpy as np

KEYS = ["code_module", "code_presentation", "id_student"]

def load_oulad(data_dir):
    info = pd.read_csv(f"{data_dir}/studentInfo.csv")
    assessments = pd.read_csv(f"{data_dir}/assessments.csv")
    student_assessment = pd.read_csv(f"{data_dir}/studentAssessment.csv")
    registration = pd.read_csv(f"{data_dir}/studentRegistration.csv")
    return info, assessments, student_assessment, registration

def assessment_features(assessments, student_assessment, cutoff):
    ass = assessments.copy()
    sa = student_assessment.merge(
        ass[["id_assessment","code_module","code_presentation","date","weight"]],
        on="id_assessment", how="left"
    )
    scheduled = (ass[ass["date"] <= cutoff]
        .groupby(["code_module","code_presentation"], as_index=False)
        .agg(scheduled_assessments=("id_assessment","nunique")))
    sub = sa[(sa["date_submitted"] <= cutoff) & (sa["date"] <= cutoff)].copy()

    def weighted(g):
        return np.average(g["score"], weights=g["weight"]) if g["weight"].sum() > 0 else g["score"].mean()

    out = (sub.groupby(KEYS).apply(lambda g: pd.Series({
        "early_assessments": g["id_assessment"].nunique(),
        "early_mean_score": g["score"].mean(),
        "early_min_score": g["score"].min(),
        "early_max_score": g["score"].max(),
        "early_score_std": g["score"].std(ddof=0),
        "early_weighted_score": weighted(g)
    }), include_groups=False).reset_index())
    out = out.merge(scheduled, on=["code_module","code_presentation"], how="left")
    out["early_assessment_attempt_rate"] = out["early_assessments"] / out["scheduled_assessments"].replace(0,np.nan)
    out["early_missed_assessments"] = out["scheduled_assessments"] - out["early_assessments"]
    return out

def engagement_features(student_vle_path, vle_path, cutoff=28, chunksize=600_000):
    vle = pd.read_csv(vle_path, usecols=["id_site","activity_type"])
    site_type = vle.set_index("id_site")["activity_type"].to_dict()
    parts = []
    for ch in pd.read_csv(student_vle_path, chunksize=chunksize,
                          usecols=["code_module","code_presentation","id_student","id_site","date","sum_click"]):
        ch = ch[(ch["date"] >= 0) & (ch["date"] <= cutoff)]
        if ch.empty:
            continue
        ch["activity_type"] = ch["id_site"].map(site_type).fillna("unknown")
        daily = (ch.groupby(KEYS+["date"], observed=True)
                   .agg(daily_clicks=("sum_click","sum"),
                        daily_sites=("id_site","nunique")).reset_index())
        parts.append(daily)

    daily = pd.concat(parts, ignore_index=True)
    daily["week"] = (daily["date"] // 7 + 1).clip(1, 8)
    out = (daily.groupby(KEYS, observed=True)
        .agg(early_total_clicks=("daily_clicks","sum"),
             early_active_days=("date","nunique"),
             early_resource_days=("daily_sites","sum"),
             early_max_daily_clicks=("daily_clicks","max"),
             early_mean_daily_clicks=("daily_clicks","mean"),
             early_click_std_daily=("daily_clicks","std")).reset_index())
    weekly = (daily.pivot_table(index=KEYS, columns="week", values="daily_clicks",
                                aggfunc="sum", fill_value=0)
              .rename(columns={i:f"week_{i}_clicks" for i in range(1,9)}).reset_index())
    out = out.merge(weekly, on=KEYS, how="outer")
    weeks = min(8, int(np.ceil(cutoff/7)))
    out[f"engagement_trend_{cutoff}"] = out[f"week_{weeks}_clicks"] - out["week_1_clicks"]
    out[f"active_day_ratio_{cutoff}"] = out["early_active_days"] / cutoff
    out[f"clicks_per_active_day_{cutoff}"] = out["early_total_clicks"] / out["early_active_days"].replace(0,np.nan)
    out[f"clicks_per_resource_day_{cutoff}"] = out["early_total_clicks"] / out["early_resource_days"].replace(0,np.nan)
    keep_weeks = [f"week_{i}_clicks" for i in range(1,weeks+1)]
    cols = list(dict.fromkeys(KEYS + [c for c in out.columns if not c.startswith("week_")] + keep_weeks))
    return out[cols].replace([np.inf,-np.inf],np.nan).fillna(0)

def build_dataset(data_dir, cutoff):
    info, assessments, student_assessment, registration = load_oulad(data_dir)
    base = info.copy()
    base["at_risk"] = base["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    base = base.merge(registration[KEYS+["date_registration"]], on=KEYS, how="left")
    base["registration_day"] = base["date_registration"].fillna(0)
    base = base.drop(columns=["date_registration","final_result"])
    eng = engagement_features(f"{data_dir}/studentVle.csv", f"{data_dir}/vle.csv", cutoff)
    ass = assessment_features(assessments, student_assessment, cutoff)
    df = base.merge(eng,on=KEYS,how="left").merge(ass,on=KEYS,how="left")
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].replace([np.inf,-np.inf],np.nan).fillna(0)
    df[f"has_early_assessment_{cutoff}"] = (df["early_assessments"] > 0).astype(int)
    return df
