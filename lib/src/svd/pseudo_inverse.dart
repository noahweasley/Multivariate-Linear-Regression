import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/svd.dart';

/// Author: Ebenmelu Ifechukwu (@noahweasley)
extension MatrixPseudoInverse on Matrix {
  /// Moore–Penrose pseudoinverse using **Golub–Reinsch SVD**.
  ///
  /// This matches LAPACK behavior:
  ///   A⁺ = V Σ⁺ Uᵀ
  Matrix pseudoInverse({
    double threshold = double.minPositive,
  }) {
    if (rows == 0 || cols == 0) {
      return transpose();
    }

    final svd = GolubReinschSVD.decompose(this);
    final results = svd.results;

    final U = results.leftSingularVectors;
    final V = results.rightSingularVectors;
    final s = results.singularValues;

    for (var i = 0; i < s.length; i++) {
      if (s[i].abs() > threshold) {
        s[i] = 1.0 / s[i];
      } else {
        s[i] = 0.0;
      }
    }

    return V.multiply(Matrix.diag(s)).multiply(U.transpose());
  }
}
