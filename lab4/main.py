import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from shared import load_data,evaluate_model

def malthus_model(t, r, N0):
    return N0 * np.exp(r * t)


def logistic_model(t, r, K, N0):
    return K / (1 + (K / N0 - 1) * np.exp(-r * t))

def main():
    path_dir = "population.xlsx"
    year, population = load_data(path_dir)

    origin_year = year[0]
    year = year - origin_year

    max_population = max(population)

    train_size = int(0.8 * len(year))
    train_years, train_population = year[:train_size], population[:train_size]
    test_years, test_population = year[train_size:], population[train_size:]

    popt_malthus, _ = curve_fit(
        malthus_model, train_years, train_population, p0=[0.01, train_population[0]]
    )
    popt_logistic, _ = curve_fit(
        logistic_model,
        train_years,
        train_population,
        p0=[0.01, max(train_population) * 2, train_population[0]],
        maxfev=10000,
    )

    future_t = np.arange(len(year), len(year) + 10)

    # 使用拟合的参数进行预测
    future_malthus = malthus_model(future_t, *popt_malthus)
    future_logistic = logistic_model(future_t, *popt_logistic)

    train_pred_malthus = malthus_model(train_years, *popt_malthus)
    test_pred_malthus = malthus_model(test_years, *popt_malthus)

    train_pred_logistic = logistic_model(train_years, *popt_logistic)
    test_pred_logistic = logistic_model(test_years, *popt_logistic)

    print("马尔萨斯模型评估结果:")
    train_malthus = evaluate_model(train_population, train_pred_malthus, max_population)
    test_malthus = evaluate_model(test_population, test_pred_malthus, max_population)
    print(f"训练集 MSE: {train_malthus[0]:.4f}, MAE: {train_malthus[1]:.4f}")
    print(f"测试集 MSE: {test_malthus[0]:.4f}, MAE: {test_malthus[1]:.4f}")

    train_logistic = evaluate_model(
        train_population, train_pred_logistic, max_population
    )
    test_logistic = evaluate_model(test_population, test_pred_logistic, max_population)
    print("\n自限模型评估结果:")
    print(f"训练集 MSE: {train_logistic[0]:.4f}, MAE: {train_logistic[1]:.4f}")
    print(f"测试集 MSE: {test_logistic[0]:.4f}, MAE: {test_logistic[1]:.4f}")

    # 绘制预测结果
    plt.figure(figsize=(12, 8))

    # 绘制历史数据
    plt.plot(year + origin_year, population, "bo-", label="history")

    # 绘制训练集拟合结果
    plt.plot(
        train_years + origin_year, train_pred_malthus, "r--", label="malthus_train"
    )
    plt.plot(
        train_years + origin_year, train_pred_logistic, "g--", label="logistic_train"
    )

    # 绘制测试集预测结果
    plt.plot(test_years + origin_year, test_pred_malthus, "r-.", label="malthus_test")
    plt.plot(test_years + origin_year, test_pred_logistic, "g-.", label="logistic_test")

    # 绘制未来十年预测结果
    future_years = np.arange(year[-1] + 1, year[-1] + 11)
    plt.plot(future_years + origin_year, future_malthus, "r-.", label="malthus_future")
    plt.plot(
        future_years + origin_year, future_logistic, "g-.", label="logistic_future"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
