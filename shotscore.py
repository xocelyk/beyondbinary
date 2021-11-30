import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from statistics import mean
import csv

# GLOBAL
FILENAME = 'ThreeData.csv'
SIGMA = 1  # can change depending on how smooth you want
# sigma is measured in inches

grid_width = 1 # has returned strange results for different values thus far; keep at 1
SIGMA = SIGMA / grid_width # to convert sigma units to inches

# bounds must be integers!
XBOUNDS = (-20, 20 + grid_width)  # the bounds (inches L/R) of our grid
(x1, x2) = XBOUNDS
YBOUNDS = (-20, 20 + grid_width)  # the bounds (inches rimDepth) of our grid
(y1, y2) = YBOUNDS
angle_bounds = [35, 55]
arcAngleProxy_to_two_dim_smooth_grid_dict = {angle: None for angle in range(angle_bounds[0], angle_bounds[1] + 1)}

# takes dataframe and returns dataframe with more parameters (we will use for our calculatioons)
def clean_data(threedata):
    df = threedata
    df = df.dropna()
    df['arcAngleProxy'] = df.apply(lambda row: get_arcAngleProxy(row), axis=1)
    df['rimDepthInches'] = df.apply(lambda row: get_rimDepthInches(row), axis=1)
    df['rimLeftRightInches'] = df.apply(lambda row: get_rimLeftRightInches(row), axis=1)
    return df


def myround(x, base=5):
    return base * round(x/base)

# appends shotScore col to dataframe
def add_shot_score(df):
    df['shotScore'] = df.apply(lambda row: get_shot_score(row),
                               axis=1)
    return df


def get_arcAngleProxy(row):
    return round(row['arcAngle'])


def get_rimDepthInches(row):
    return row['rimDepth'] * 12


def get_rimLeftRightInches(row):
    return row['rimLeftRight'] * 12

# calculates shotScore for individual shot
def get_shot_score(row):
    angle = row['arcAngleProxy']
    if angle not in arcAngleProxy_to_two_dim_smooth_grid_dict.keys():
        return None
    two_dim_gaussian = arcAngleProxy_to_two_dim_smooth_grid_dict[angle]
    score = gaussian_to_score(row, two_dim_gaussian)
    return score

# returns shotScore as a function of the two-dimensional gaussian-smoothed grid
def gaussian_to_score(row, smooth_grid):
    x = myround(row['rimLeftRightInches'], base=grid_width)
    y = myround(row['rimDepthInches'], base=grid_width)
    x_idx = int(1/grid_width * x)
    y_idx = int(1/grid_width * y)
    if not (XBOUNDS[0] <= x_idx < XBOUNDS[1] and YBOUNDS[0] <= y_idx < YBOUNDS[1]):
        return None
    return smooth_grid[x_idx][y_idx]

# returns the shotScore grid (probability_make_grid) BEFORE smoothing
def get_gridded_shots(angle, df, xbounds, ybounds):
    num_x_cells = int(1/grid_width * (xbounds[1] - xbounds[0]))
    num_y_cells = int(1/grid_width * (ybounds[1] - ybounds[0]))
    grid_total = np.ones((num_x_cells, num_y_cells)) # +1 smoothing, eliminates div error too
    grid_makes = np.zeros((num_x_cells, num_y_cells))

    for idx, row in df.iterrows():
        if row['arcAngleProxy'] == angle:
            x = row['rimLeftRightInches']
            y = row['rimDepthInches']
            array_x_idx = int(1/grid_width) * int(myround(x, base=grid_width) - xbounds[0]) # necessary that bounds be whole numbers
            array_y_idx = int(1/grid_width) * int(myround(y, base=grid_width) - ybounds[0])
            if array_x_idx < 0 or array_x_idx >= xbounds[1] - xbounds[0] or array_y_idx < 0 or array_y_idx >= ybounds[
                1] - ybounds[0]:
                continue
            grid_total[array_y_idx][array_x_idx] += 1
            if row['outcome']:
                grid_makes[array_y_idx][array_x_idx] += 1

    total_shots = 0
    total_makes = 0

    for row in grid_total:
        total_shots += sum(row)
    for row in grid_makes:
        total_makes += sum(row)
    total_shots -= 1 * int( 1/grid_width * (xbounds[1] - xbounds[0])) ** 2 # to account for initializing with +1 smoothing
    probability_make_grid = np.divide(grid_makes, grid_total)
    return probability_make_grid

# returns the shotScore grid for a given angle and appends to the dataframe
def get_smooth_gridded_shots(angle, df, xbounds, ybounds):
    # unsmoothed grid
    probability_make_grid = get_gridded_shots(angle=angle, df=df, xbounds=xbounds, ybounds=ybounds)
    smooth_array = smooth_grid(probability_make_grid, sigma=SIGMA)
    smooth_array_df = pd.DataFrame(smooth_array, index=[i*grid_width for i in range(int(1/grid_width * xbounds[0]), int(1/grid_width * xbounds[1]))],
                                   columns=[i*grid_width for i in range(int(1/grid_width * ybounds[0]), int(1/grid_width * ybounds[1]))])
    return smooth_array_df

# performs the Gaussian smoothing for shotScore
def smooth_grid(grid, sigma):
    smooth_array = gaussian_filter(grid, sigma=sigma)
    return smooth_array

# identifies average shotScore and 3PAV for each player
# only for use after performing the R regression once shotScore is calculated for each shot
def get_player_shot_score(read_filename, write_filename):
    df = pd.read_csv(read_filename)
    player_shotscore = {}
    player_outcome = {}
    player_adjusted_accuracy = {}
    id_to_name = {}

    for idx, row in df.iterrows():
        if str(row['shotScore']) == 'nan':  # drop null values
            continue
        if row['shooterId'] not in id_to_name.keys():
            id_to_name[row['shooterId']] = [row['firstName'], row['lastName']]
        if row['shooterId'] not in player_shotscore.keys():
            player_shotscore[row['shooterId']] = [row['shotScore']]
            # print(round(row['shotScore']*100, 1))
        else:
            player_shotscore[row['shooterId']].append(row['shotScore'])
        if row['shooterId'] not in player_outcome.keys():
            player_outcome[row['shooterId']] = [int(row['outcome'])]
        else:
            player_outcome[row['shooterId']].append(int(row['outcome']))
        if row['shooterId'] not in player_adjusted_accuracy.keys():
            player_adjusted_accuracy[row['shooterId']] = [row['adjustedAccuracy']]
        else:
            player_adjusted_accuracy[row['shooterId']].append(row['adjustedAccuracy'])

    player_dict = {}

    for key, val in id_to_name.items():
        player_dict[key] = [val[0], val[1]] + [mean(player_shotscore[key])] + [mean(player_outcome[key])] + [
            mean(player_adjusted_accuracy[key])] + [len(player_outcome[key])]

    with open(write_filename, 'w') as f:
        writer = csv.writer(f)
        header = ['shooterId', 'firstName', 'lastName', 'shotScore', 'outcome', 'adjustedAccuracy', 'n']
        writer.writerow(header)
        for key, value in player_dict.items():
            writer.writerow([key] + value)

# saves the smoother shotScore grids to csv files
def write_smooth_grids(df):
    for angle in arcAngleProxy_to_two_dim_smooth_grid_dict.keys():
        arcAngleProxy_to_two_dim_smooth_grid_dict[angle] = get_smooth_gridded_shots(angle=angle, df=df, xbounds=XBOUNDS, ybounds=YBOUNDS)
        print('Angle: ' + str(angle))
        print('Max Smooth Accuracy by rimDepth: \n' + str(round(np.amax(arcAngleProxy_to_two_dim_smooth_grid_dict[angle]), 3)))
        print()
        filename = 'kernel_smooth_2d/' + str(angle) + '.csv'
        np.savetxt(filename, arcAngleProxy_to_two_dim_smooth_grid_dict[angle], delimiter=",")

def main():
    threedata = clean_data(pd.read_csv(FILENAME))
    write_smooth_grids(threedata)
    add_shot_score(threedata)
    threedata.to_csv('threedata_shotscore.csv')

    # get_player_shot_score can only be executed after the R script to generate three_data_shotscore_2.csv
    # is run; this R script does the regression of ShotScore on shot factors and writes the predicted results
    # for each attempt to three_data_shotscore_2.csv
    # for this reason the below is commented out
    # another approach would be to perform the regression in python

    # get_player_shot_score('threedata_shotscore_2.csv', 'player_shotscore.csv')


if __name__ == '__main__':
    main()
