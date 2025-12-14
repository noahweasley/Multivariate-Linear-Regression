import 'dart:math' as math;

import 'package:ml_linalg/matrix.dart';

/// Author: @noahweasley
/// Utility extensions for Matrix
extension MatrixExtension on Matrix {
  /// Computes the pseudo-inverse of the matrix
  Matrix pseudoInverse() {
    final rows = rowCount;
    final cols = columnCount;

    if (rows >= cols) {
      final xt = transpose();
      final xtx = xt.multiply(this);
      return xtx.inverse().multiply(xt);
    } else {
      final xxT = multiply(transpose());
      return transpose().multiply(xxT.inverse());
    }
  }

  /// Returns the diagonal elements of the matrix as a list
  List<double> diagonal() {
    final minDim = math.min(rowCount, columnCount);
    final diagonal = <double>[];

    for (var i = 0; i < minDim; i++) {
      diagonal.add(this[i][i]);
    }

    return diagonal;
  }
}
