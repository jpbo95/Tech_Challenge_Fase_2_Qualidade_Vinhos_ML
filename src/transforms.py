from sklearn.preprocessing import StandardScaler

def split_features_target(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def standardize_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled