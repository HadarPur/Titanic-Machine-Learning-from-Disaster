def sex_survival_counts(name, X, y, sex_col):
    sex = X[:, sex_col]
    female = sex == 1
    male = sex == 0
    survived = y == 1
    print(
        f"{name:<16}| "
        f"female survived {int((female & survived).sum()):3d}, "
        f"not {int((female & ~survived).sum()):3d} | "
        f"male survived {int((male & survived).sum()):3d}, "
        f"not {int((male & ~survived).sum()):3d}"
    )