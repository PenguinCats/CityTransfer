import folium


class VisualizationTool(object):
    def __init__(self, args):
        self.args = args
        self.visual_map = folium.Map(
            tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            attr='default', location=[32.041544, 118.767413], zoom_start=12)

    def draw_map(self, real_bounds, pred_circles, pred_back_circles, other_shops_bounds):
        for other_bound in real_bounds:
            folium.Rectangle(
                bounds=other_bound,
                color='#ff0000',
                fill=True,
                fill_color='#ff0000',
                fill_opacity=0.2
            ).add_to(self.visual_map)
        for pred_circle in pred_circles:
            folium.Circle(
                location=pred_circle,
                # radius=self.args.circle_size,
                radius=250,
                color='#004080',
                fill=True,
                fill_color='#004080',
                fill_opacity=0.2
            ).add_to(self.visual_map)
        for pred_circle in pred_back_circles:
            folium.Circle(
                location=pred_circle,
                # radius=self.args.circle_size,
                radius=250,
                color='#FFFF00',
                fill=True,
                fill_color='#FFFF00',
                fill_opacity=0.2
            ).add_to(self.visual_map)
        for other_bound in other_shops_bounds:
            folium.Polygon(
                locations=other_bound,
                color='#00FF00',
                fill=True,
                fill_color='#00FF00',
                fill_opacity=0.2
            ).add_to(self.visual_map)
        self.visual_map.save("mp_trick_feature_removed.html")


if __name__ == '__main__':
    v = VisualizationTool(None)
    ss = [[[31.980166999999998, 118.757457], [31.975167, 118.757457], [31.975167, 118.762457], [31.980166999999998, 118.762457], [31.980166999999998, 118.757457]]]
    tg = [[31.997666999999993, 118.764957], [32.062666999999976, 118.77495699999999]]
    v.draw_map(ss, tg, tg, tg)
