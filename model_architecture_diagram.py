from graphviz import Digraph

dot = Digraph(comment='Lung Cancer Detection Architecture')

dot.attr(rankdir='LR', size='8,5')

dot.node('A', 'CT Scan Image\n(224x224)')
dot.node('B', 'Preprocessing\nNormalization\nAugmentation')
dot.node('C', 'ResNet-18 Backbone')
dot.node('D', 'CBAM Attention Module\n(Channel + Spatial)')
dot.node('E', 'Fully Connected Layer')
dot.node('F', 'Temperature Scaling\nCalibration')
dot.node('G', 'Prediction\nBenign / Malignant')

dot.edges(['AB','BC','CD','DE','EF','FG'])

dot.render('model_architecture', format='png', cleanup=True)

print("Architecture diagram saved as model_architecture.png")