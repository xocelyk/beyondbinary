from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns

directory = '/Users/kylecox/Documents/UT/SDS 379R/beyondbinary/Percentage by Depth/'
plt.style.use('fivethirtyeight')



def plot_pct_vs_vol_over_time():
	# data from basketball reference
	season = np.array([i for i in range(2000, 2022)])
	pct = np.array([.353, .354, .354, .349, .347, .356, .358, .358, .362, .367, .355, .358, .349, .359, .360, .350, .354, .358, .362, .355, .358, .367])
	vol = np.array([13.7, 13.7, 14.7, 14.7, 14.9, 15.8, 16, 16.9, 18.1, 18.1, 18.1, 18.0, 18.4, 20, 21.5, 22.4, 24.1, 27.0, 29.0, 32.0, 34.1, 34.6])
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

def plot_best_rim_depth_by_angle():
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
	plt.xlabel('Arc Angle (degrees)', fontsize=15)
	plt.ylabel('Rim Depth (in.)', fontsize=15)
	plt.title('Ideal Rim Depth from Launch Angle', fontsize=15)
	plt.subplots_adjust(bottom=.15)
	plt.subplots_adjust(left=.15)
	plt.savefig('viz/best-rim-depth-by-angle.png')
	plt.clf()


def grid_heatplots():
	for angle in [38, 42, 46, 50]:
		print(angle)
		# i = angle - 38 + 1
		data = []
		df = pd.read_csv('kernel_smooth_2d/' + str(angle) + '.csv', names = [str(i) for i in range(-20, 20+1)] )
		df.index = [str(i) for i in range(-20, 20 + 1)]
		# df = df.iloc[::-1]

		small_df = df[[str(i) for i in range(-10, 10 + 1)]]
		small_df = small_df.loc[[str(i) for i in range(-10, 10 + 1)]]

		plt.figure(figsize=(10*6273/5500,10), dpi=1000)
		ax = sns.heatmap(small_df, linewidth=0.5, annot=False)
		plt.gca().invert_yaxis()
		plt.xlabel('Rim Left-Right (in.)', fontsize=15)
		plt.ylabel('Rim Depth (in.)', fontsize=15)
		# add circle with radius 9 to plot
		plt.gca().add_artist(plt.Circle((0 + 10.5, 0 + 10.5), radius=9, color='orange', fill=False, linewidth=2))
		plt.title('Probability Make by x-y Placement, Angle: ' + str(angle) + ' degrees', fontsize=15)
		plt.subplots_adjust(bottom=.15)
		plt.subplots_adjust(left=.15)
		plt.savefig('viz/kernel-smooth-2d-2/' + str(angle) + '.png')
		plt.clf()

	df = pd.read_csv('threedata_shotscore_2.csv')
	# plot a histogram of arcAngle in the df
	plt.hist(df['arcAngle'], bins=100)
	plt.xlim([30, 60])
	plt.xlabel('Shot Angle (degrees)')
	plt.ylabel('Frequency')
	plt.savefig('viz/arc_angle_histogram.png')


# plot average shot score per year and average outcome per year on same axis
def plot_shooting_percentage_and_shotscore_over_seasons(df):
	seasons = [i for i in range(2012, 2020)]
	pct_series = []
	ss_series = []
	for season in seasons:
		pct_series.append(df[df['season'] == season]['outcome'].mean())
		ss_series.append(df[df['season'] == season]['shotScore'].mean())
	## format ss_series as percentage
	ss_series = [i*100 for i in ss_series]
	# format pct_series as percentage
	pct_series = [i*100 for i in pct_series]
	# plot pct_series and ss_series on seasons axis
	plt.plot(seasons, pct_series, label='Shooting Percentage')
	plt.plot(seasons, ss_series, label='Shot Score')
	# make y-axis a percentage
	plt.ylim(30, 40)
	plt.title('Shooting Percentage and Shot Score (* 100) Over Seasons')
	plt.xlabel('Season')
	plt.ylabel('Shot Score * 100 /Three-Point Percentage')
	plt.legend()
	plt.savefig('viz/shot-score-and-pct-over-seasons.png')
	plt.show()
	plt.clf()

def main():
	df = pd.read_csv('threedata_shotscore_2.csv')
	plot_shooting_percentage_and_shotscore_over_seasons(df)

if __name__ == '__main__':
	main()


