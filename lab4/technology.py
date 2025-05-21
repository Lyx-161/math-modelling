import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from shared import load_data,evaluate_model


def technology_model(t, r, K, a, N0):
    K_t=K*np.exp(a*t)
    return K_t / (1 + (K_t / N0 - 1) * np.exp(-r * t))

def technology_model_log(t, r, K, a, N0):
    K_t=K*a*np.log(t+np.e)
    return K_t / (1 + (K_t / N0 - 1) * np.exp(-r * t))

def delay_model(t,r,tao,N0):
    return N0*np.exp(r*(t-tao)*(t-tao)/2)

def main():
    path_dir = "population.xlsx"
    year, population = load_data(path_dir)

    origin_year = year[0]
    year = year - origin_year

    max_population = max(population)

    train_size = int(0.8 * len(year))
    train_years, train_population = year[:train_size], population[:train_size]
    test_years, test_population = year[train_size:], population[train_size:]

    model = delay_model

    popt_logistic, _ = curve_fit(
        model,
        train_years,
        train_population,
        # p0=[0.01, max(train_population) * 2,0.01, train_population[0]]，
        p0=[0.01,2,train_population[0]],
        # maxfev=10000,
    )

    future_t = np.arange(len(year), len(year) + 10)

    # 使用拟合的参数进行预测
    future= model(future_t, *popt_logistic)

    train_pred = model(train_years, *popt_logistic)
    test_pred = model(test_years, *popt_logistic)

    train = evaluate_model(
        train_population, train_pred, max_population
    )
    test= evaluate_model(test_population, test_pred, max_population)
    print("\n技术进步模型评估结果:")
    print(f"训练集 MSE: {train[0]:.4f}, MAE: {train[1]:.4f}")
    print(f"测试集 MSE: {test[0]:.4f}, MAE: {test[1]:.4f}")

    # 绘制预测结果
    plt.figure(figsize=(12, 8))

    # 绘制历史数据
    plt.plot(year + origin_year, population, "bo-", label="history")
    plt.plot(train_years + origin_year, train_pred, "g--", label="train")
    plt.plot(test_years + origin_year, test_pred, "g-.", label="test")
    future_years = np.arange(year[-1] + 1, year[-1] + 11)
    plt.plot(future_years + origin_year, future, "g-.", label="future")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
