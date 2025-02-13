# sys.path.insert(1, '..')
# from feature_selection import feature_selection_anova, feature_selection_pca, feature_selection_black_box_rf,feature_selection_wd,feature_selection_ks,feature_selection_ps, load_data

def pipeline(df,mod):
    PCA_transform = None
    # Pipeline
    df = feature_selection_ps(df, percentile=10)
    df, nf = feature_selection_black_box_rf(df, mod, auto_select_kp="ACC", show_graph=True)

    # PCA BELOW
    pre_pca_feature_names = df.columns  # Uncomment line below for PCA. Keep commented to disable PCA.
    # df, PCA_transform = feature_selection_pca(df, num_features=10)
    return df, pre_pca_feature_names, PCA_transform