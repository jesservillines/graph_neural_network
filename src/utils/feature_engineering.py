def engineer_features(df):
    """Create advanced features for the model"""
    df = df.copy()
    
    # Age-related interactions
    df['age_comorbidity_interaction'] = df['admit_age'] * df['age_adj_comorbidity_score']
    
    # Injury severity composite
    df['injury_severity_score'] = (
        df['brain_injury_mild/_moderate'].astype(float) * 1 +
        df['brain_injury_severe'].astype(float) * 2
    )
    
    # Medical complexity index
    complexity_columns = [
        'diabetes', 'hypertension', 'heart_disease',
        'neurological_disorder', 'psychiatric_disorder'
    ]
    df['medical_complexity_index'] = df[complexity_columns].sum(axis=1)
    
    return df