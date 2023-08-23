import pickle

with open('cameraLaser_points.pkl', 'rb') as fp:
    data = pickle.load(fp)

print(data)