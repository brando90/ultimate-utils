import scipy.integrate as integrate

def functional_norm_diff(f_target, f_predicted, lb=-1, ub=1, p=2):
    point_diff = lambda x: (f_target(x) - f_predicted(x))**2
    norm, abs_err = integrate.quad(point_diff, a=lb, b=ub
    return norm**2, abs_err

if __name__ == '__main__':
    norm = functional_norm_diff(lambda x: x, lambda x: x**2)
    print(f'norm = {norm}')