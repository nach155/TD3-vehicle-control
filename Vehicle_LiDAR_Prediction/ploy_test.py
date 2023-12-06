from  shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt

def main():
    # fig = plt.figure(1, dpi=90)
    # ax = fig.add_subplot(121)
    ext = [(0, 0), (0, 2), (2, 2), (2, 0)]
    asd = [(1,1),(1,3),(3,3),(3,1)]
    # point = Point(1,1)
    line = LineString([(-1,-1),(1,1)])
    polygon = Polygon(ext)
    a = Polygon(asd)
    x = polygon.exterior.coords[:-1]
    print(x)
    # plt.show()
    # print(polygon.contains(point))
    print(polygon.difference(a).exterior.coords[:-1])
###############################33
if __name__ == '__main__':
    main()