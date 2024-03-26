import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
	flags = argparse.ArgumentParser(description='ft_linear_regression model trainer')
	flags.add_argument('dataset', type=str, help='csv dataset file')
	flags.add_argument('-o', '--output', type=str, default='thetas.csv', help='output csv file')
	flags.add_argument('-r', '--rate', type=float, default=0.1, help='learning rate')
	flags.add_argument('-e', '--epochs', type=int, default=1000, help='learning epochs')
	flags.add_argument('-a', '--auto', action="store_true", default=False, help='auto stop the learning epochs when the mse is not improving')
	flags.add_argument('-p', '--plot', action="store_true", default=False, help='plots the regression')
	flags.add_argument('-H', '--history', action="store_true", default=False, help='plots the history of thetas')
	flags.add_argument('-m', '--mse', action="store_true", default=False, help='plots the history of the mean squared error')
	args = flags.parse_args()

	if args.rate <= 0 or args.rate > 1:
		raise ValueError('Invalid learning rate')
	if args.epochs < 1:
		raise ValueError('Invalid number of epochs')
	return args


def standardize(data):
	return (data - np.mean(data)) / np.std(data)


def estimate_price(km, theta0, theta1):
	return theta0 + (theta1 * km)


history0, history1 = [], []
history_mse = []
def train(learning_rate, epochs, km, price, auto_stop):
	theta0 = 0
	theta1 = 0
	m = len(km)
	for i in range(epochs):
		tmp0 = learning_rate * (1 / m) * np.sum(estimate_price(km, theta0, theta1) - price)
		tmp1 = learning_rate * (1 / m) * np.sum((estimate_price(km, theta0, theta1) - price) * km)
		theta0 -= tmp0
		theta1 -= tmp1
		# History
		history0.append(theta0)
		history1.append(theta1)
		# Auto stop
		mse = np.sum((estimate_price(km, theta0, theta1) - price) ** 2) / m
		history_mse.append(mse)
		if auto_stop and len(history_mse) > 1 and (
			np.abs((mse - history_mse[-2]) / ((mse + history_mse[-2]) / 2)) < 1e-7):
			break
		
	return theta0, theta1


def main():
	args = get_args()

	# Data
	data = pd.read_csv(args.dataset)
	data = data.sort_values(by='km', ignore_index=True)
	data_km = np.array(data['km'])
	data_price = np.array(data['price'])
	km = standardize(data_km)
	price = standardize(data_price)

	# Train
	learning_rate = args.rate
	epochs = args.epochs
	auto_stop = args.auto
	theta0, theta1 = train(learning_rate, epochs, km, price, auto_stop)
	theta1 = theta1 * np.std(data_price) / np.std(data_km)
	theta0 = theta0 * np.std(data_price) + np.mean(data_price) - theta1 * np.mean(data_km)

	# Output
	with open(args.output, 'w') as f:
		f.write(f'theta0,theta1\n{theta0},{theta1}\n')

	if args.plot:
		plt.scatter(data_km, data_price, color='blue')
		plt.plot(data_km, estimate_price(data_km, theta0, theta1), color='red')
		plt.xlabel('km')
		plt.ylabel('price')
		plt.title('ft_linear_regression')
		plt.show()

	if args.history:
		hst, (hist0, hist1) = plt.subplots(2)
		hst.suptitle("History of Thetas and and Thetas' Difference during training")
		hist0.plot(list(range(1, len(history0)+1)), history0, color='purple')
		hist1.plot(list(range(1, len(history1)+1)), history1, color='lime')
		hist0.set_title("Theta0")
		hist1.set_title("Theta1")
		plt.show()
	
	if args.mse:
		plt.plot(list(range(1, len(history_mse)+1)), history_mse, color='orange')
		plt.title("Mean Squared Error during training")
		plt.show()


if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		print('Error:', e)
