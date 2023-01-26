# -

def example_test_percentages_of_y_explained_by_x_synthetic_variable():
    """
    - plan
        - generate x_i using N(mu_i,std_i)
        - generate some true coeff_i for each
        - y_true = sum(coeff_i * x_i) + 1/4 * N(sum_i(mu_i), sum_i(std_i))
        - then play around with
            - coeff_i, how much do percentage of y explained by x_i change?
            - mu_i, how much do percentage of y explained by x_i change?
            - std_i, how much do percentage of y explained by x_i change?
            - but have a concrete hypothesis first
    """


def example_test_explaining_variance_of_depedent_variable_from_multiple_indepdent_variables():
    """
    I want to break down the amount or percentage of variance each of the multiple independent variable explains of the single dependent variable, how do I do it in python with an example?
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Create a sample dataset
    data = {'x1': [1, 2, 3, 4, 5],
            'x2': [2, 3, 4, 5, 6],
            'y': [3, 5, 7, 9, 11]}
    df = pd.DataFrame(data)

    # Create the linear regression object
    reg = LinearRegression()

    # Fit the model using the independent variables x1 and x2
    reg.fit(df[['x1', 'x2']], df['y'])

    # Print the R-squared for each independent variable
    print("R-squared for x1:", reg.score(df[['x1', 'x2']], df['y']))

    # Create a new dataframe with only x2
    df_x2 = df[['x2']]

    # fit the model with only x2
    reg.fit(df_x2, df['y'])

    # Print the R-squared for x2
    print("R-squared for x2:", reg.score(df_x2, df['y']))


# -- run it

if __name__ == '__main__':
    import time

    start = time.time()
    # - run it
    example_test_explaining_variance_of_depedent_variable_from_multiple_indepdent_variables()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
