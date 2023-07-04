import pandas as pd
import matplotlib.pyplot as plt

class VisualizationStrategy:
    def visualize(self, data):
        pass

class BarChartStrategy(VisualizationStrategy):
    def visualize(self, data):
        plt.figure(figsize=(10, 6))
        plt.bar(data.index, data.values)
        plt.xlabel(data.index.name)
        plt.ylabel("Count")
        plt.title("Bar Chart - {}".format(data.index.name))
        plt.xticks(rotation=90)
        plt.show()

class PieChartStrategy(VisualizationStrategy):
    def visualize(self, data):
        plt.figure(figsize=(8, 8))
        plt.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Pie Chart - {}".format(data.index.name))
        plt.show()

class LinePlotStrategy(VisualizationStrategy):
    def visualize(self, data):
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data.values)
        plt.xlabel(data.index.name)
        plt.ylabel(data.values.name)
        plt.title("Line Plot - {} vs. {}".format(data.index.name, data.values.name))
        plt.show()

class DataVisualizer:
    def __init__(self, strategy):
        self.strategy = strategy

    def visualize_data(self, data):
        self.strategy.visualize(data)

# Usage example
df = pd.read_csv("test.csv")
data = df["Donor_Category"].value_counts()

bar_chart_strategy = BarChartStrategy()
visualizer = DataVisualizer(bar_chart_strategy)
visualizer.visualize_data(data)

pie_chart_strategy = PieChartStrategy()
visualizer = DataVisualizer(pie_chart_strategy)
visualizer.visualize_data(data)
