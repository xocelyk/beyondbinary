import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from statistics import mean
import csv

## GLOBAL
FILENAME = 'ThreeData.csv'
SIGMA = 1  # can change depending on how smooth you want
XBOUNDS = (-20, 20 + 1)  # the ILENAbounds (inches L/R) of our grid
(x1, x2) = XBOUNDS
YBOUNDS = (-20, 20 + 1)  # the bounds (inches rimDepth) of our grid
(y1, y2) = YBOUNDS
angle_bounds = [38, 53]
grid_width = 1  # not in use yet because we use the round (to integer) function
arcAngleProxy_to_two_dim_smooth_grid_dict = {angle: None for angle in range(angle_bounds[0], angle_bounds[1] + 1)}


# takes dataframe and returns dataframe with more objects
def clean_data(threedata):
    df = threedata
    df = df.dropna()
    df['arcAngleProxy'] = df.apply(lambda row: get_arcAngleProxy(row), axis=1)
    df['rimDepthInches'] = df.apply(lambda row: get_rimDepthInches(row), axis=1)
    df['rimLeftRightInches'] = df.apply(lambda row: get_rimLeftRightInches(row), axis=1)
    return df


def get_shot_score(df):
    df['shotScore'] = df.apply(lambda row: add_shot_score(row, arcAngleProxy_to_two_dim_smooth_grid_dict),
                               axis=1)
    return df


def get_arcAngleProxy(row):
    return round(row['arcAngle'])


def get_rimDepthInches(row):
    return row['rimDepth'] * 12


def get_rimLeftRightInches(row):
    return row['rimLeftRight'] * 12


def add_shot_score(row, angle_to_smooth_grid_dict):
    angle = row['arcAngleProxy']
    if angle not in angle_to_smooth_grid_dict.keys():
        return None
    two_dim_gaussian = angle_to_smooth_grid_dict[angle]
    score = gaussian_to_score(row, two_dim_gaussian)
    return score


def gaussian_to_score(row, smooth_grid):
    x = round(row['rimLeftRightInches'])
    y = round(row['rimDepthInches'])
    if not (XBOUNDS[0] <= x < XBOUNDS[1] and YBOUNDS[0] <= y < YBOUNDS[1]):
        return None
    return smooth_grid[x][y]


def get_gridded_shots(angle, df, xbounds, ybounds):
    grid_total = np.ones((xbounds[1] - xbounds[0], ybounds[1] - ybounds[0]))
    grid_makes = np.zeros((xbounds[1] - xbounds[0], ybounds[1] - ybounds[0]))

    for idx, row in df.iterrows():
        if row['arcAngleProxy'] == angle:
            x = row['rimLeftRightInches']
            y = row['rimDepthInches']
            array_x_idx = round(x) - xbounds[0]
            array_y_idx = round(y) - ybounds[0]
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
    total_shots -= 1 * (xbounds[1] - xbounds[0]) ** 2

    print('Total Makes: ', end='')
    print(total_makes, end='')
    print(', Total Shots: ', end='')
    print(total_shots, end=', ')

    probability_make_grid = np.divide(grid_makes, grid_total)
    print('Max Accuracy: ' + str(round(np.amax(probability_make_grid), 3)), end=', ')
    return probability_make_grid


def get_smooth_gridded_shots(angle, df, xbounds, ybounds):
    probability_make_grid = get_gridded_shots(angle=angle, df=df, xbounds=xbounds, ybounds=ybounds)
    smooth_array = smooth_grid(probability_make_grid, sigma=SIGMA)
    smooth_array_df = pd.DataFrame(smooth_array, index=[i for i in range(xbounds[0], xbounds[1])],
                                   columns=[i for i in range(ybounds[0], ybounds[1])])
    return smooth_array_df


def smooth_grid(grid, sigma):
    smooth_array = gaussian_filter(grid, sigma=sigma)
    return smooth_array


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


def write_smooth_grids(df):
    for angle in arcAngleProxy_to_two_dim_smooth_grid_dict.keys():
        arcAngleProxy_to_two_dim_smooth_grid_dict[angle] = get_smooth_gridded_shots(angle=angle, df=df,
                                                                                    xbounds=XBOUNDS,
                                                                                    ybounds=YBOUNDS)
        print('Max Smooth Accuracy by rimDepth: \n' + str(round(np.amax(arcAngleProxy_to_two_dim_smooth_grid_dict[angle]), 3)))
        filename = 'kernel_smooth_2d/' + str(angle) + '.csv'
        np.savetxt(filename, arcAngleProxy_to_two_dim_smooth_grid_dict[angle], delimiter=",")


def main():
    threedata = clean_data(pd.read_csv(FILENAME))
    write_smooth_grids(threedata)
    add_shot_score(threedata)
    threedata.to_csv('threedata_shotscore.csv')
    get_player_shot_score('threedata_shotscore.csv', 'player_shotscore.csv')


if __name__ == '__main__':
    main()
