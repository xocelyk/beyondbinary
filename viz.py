from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')

season = np.array([i for i in range(2000, 2022)])
pct = np.array([.353, .354, .354, .349, .347, .356, .358, .358, .362, .367, .355, .358, .349, .359, .360, .350, .354, .358, .362, .355, .358, .367])
vol = np.array([13.7, 13.7, 14.7, 14.7, 14.9, 15.8, 16, 16.9, 18.1, 18.1, 18.1, 18.0, 18.4, 20, 21.5, 22.4, 24.1, 27.0, 29.0, 32.0, 34.1, 34.6])

np.random.seed(19680801)


plt.plot(season, pct)
plt.xlabel('Season', fontsize=15)
plt.ylabel('Three-Point Percentage', fontsize=15)
plt.ylim(0, .5)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=.15)
plt.savefig('viz/three-point-pct-over-time.png')
plt.clf()

plt.plot(season, vol)
plt.xlabel('Season', fontsize=15)
plt.ylabel('Mean Three-Point Attempts per Game', fontsize=15)
plt.ylim(0, 40)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=.15)
plt.savefig('viz/three-point-attempts-per-game.png')
plt.clf()

import csv
directory = '/Users/kylecox/Documents/UT/SDS 379R/beyondbinary/Percentage by Depth/'

# fig = plt.figure(figsize=(40, 40))
# columns = 5
# rows = 3
angles = []
best_depth = []
for angle in range(38, 52 + 1):
	angles.append(angle)
	res = {}
	# i = angle - 38 + 1
	filename = directory + str(angle) + '.csv'
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			res[row[1]] = float(row[2])
		max_key = float(max(res, key=res.get))
		best_depth.append(round(max_key,1))
print(angles)
print(best_depth)
plt.scatter(np.array(angles), np.array(best_depth))
plt.xlabel('Rim Depth', fontsize=15)
plt.ylabel('Probability Make', fontsize=15)
plt.title('Ideal Rim Depth from Launch Angle', fontsize=15)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=.15)
plt.savefig('viz/best-rim-depth-by-angle.png')
plt.clf()


import seaborn as sns
for angle in [40, 45, 50]:
	print(angle)
	# i = angle - 38 + 1
	data = []
	df = pd.read_csv('/Users/kylecox/Documents/UT/SDS 379R/beyondbinary/kernel_smooth_2d/' + str(angle) + '.csv', names = [str(i) for i in range(-20, 20+1)] )
	df.index = [str(i) for i in range(-20, 20 + 1)]
	# df = df.iloc[::-1]

	small_df = df[[str(i) for i in range(-10, 10 + 1)]]
	small_df = small_df.loc[[str(i) for i in range(-10, 10 + 1)]]

	# with open('/Users/kylecox/Documents/UT/SDS 379R/beyondbinary/kernel_smooth_2d/' + str(angle) + '.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	for row in reader:
	# 		data.append(row)
	# data = np.array(data, dtype='float')
	# print(data.size)
	# img = 
	plt.figure(figsize=(10,10), dpi=1000)
	ax = sns.heatmap(small_df, linewidth=0.5, annot=False)
	plt.gca().invert_yaxis()
	plt.xlabel('Rim Left-Right (in.)', fontsize=15)
	plt.ylabel('Rim Depth (in.)', fontsize=15)
	plt.title('Probability Make by x-y Placement, Angle: ' + str(angle) + ' degrees', fontsize=15)
	plt.subplots_adjust(bottom=0.15)
	plt.subplots_adjust(left=.15)
	plt.savefig('viz/kernel-smooth-2d-2/' + str(angle) + '.png')
	# fig.add_subplot(rows, columns, i)
	plt.clf()

    # img = np.random.randint(10, size=(h,w))
    # fig.add_subplot(rows, columns, i)
    # plt.scatter()

import cv2
w = 10
h = 10
fig = plt.figure(figsize=(10, 10))
columns = 3
rows = 5
i = 1
for angle in range[40, 45, 50]:
	im = cv2.imread('/iz/kernel-smooth-2d-2/' + str(angle) + '.png')
	fig.add_subplot(rows, columns, i)
	i += 1
plt.savefig('viz/kernel-grid')

