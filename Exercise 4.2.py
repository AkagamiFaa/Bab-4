#Exercise 4.2
# Modify Example 4.2 to have three inputs instead of two.
from sklearn.neural_network import MLPClassifier

# Three inputs
X = [[0., 0., 0.], [1., 1., 1.], [0., 1., 0.], [1., 0., 1.]]
y = [0, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print([coef.shape for coef in clf.coefs_])
