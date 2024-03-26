import numpy as np
import pandas as pd


def main():
	# Read Thetas
	thetas = pd.read_csv('thetas.csv')
	thetas = np.array(thetas)
	if thetas.size == 0:
		raise Exception('Thetas not found')
	thetas = thetas[0, :]
	theta0, theta1 = thetas[0], thetas[1]
	if theta0 == 0 and theta1 == 0:
		raise Exception('Thetas not set')

	# Get Kilometers
	km = input('Enter kilometers: ')
	if not km.isdigit():
		raise Exception('Invalid input')
	km = int(km)

	# Predict Price
	estimated_price = theta0 + (theta1 * km)
	if estimated_price <= 0:
		raise Exception('Invalid price')
	print(f'Estimated price: {estimated_price}')


if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		print(f'Error: {e}')
