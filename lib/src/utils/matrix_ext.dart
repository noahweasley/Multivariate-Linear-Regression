import 'package:ml_linalg/matrix.dart';

///
extension PseudoInverse on Matrix {
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
}
