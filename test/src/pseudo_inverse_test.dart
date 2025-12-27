import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/pseudo_inverse.dart';
import 'package:test/test.dart';

void main() {
  group('Matrix pseudoInverse (SVD-based)', () {
    test('identity matrix returns itself', () {
      final I = Matrix.identity(3);
      final pInv = I.pseudoInverse();

      expect(pInv.toList(), equals(I.toList()));
    });

    test('tall rectangular matrix (3x2)', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);

      final AInverse = A.pseudoInverse();

      expect(AInverse.rows, equals(2));
      expect(AInverse.cols, equals(3));

      // Moore–Penrose
      final reconstructed = AInverse.multiply(A).multiply(AInverse);

      for (var i = 0; i < AInverse.rows; i++) {
        for (var j = 0; j < AInverse.cols; j++) {
          expect(
            (reconstructed.get(i, j) - AInverse.get(i, j)).abs() < 1e-10,
            isTrue,
          );
        }
      }
    });

    // TODO: Fix test case
    test('empty matrix returns transpose', () {
      final A = Matrix.zeros(0, 0);
      final AInverse = A.pseudoInverse();

      expect(AInverse.rows, equals(0));
      expect(AInverse.cols, equals(0));
    });

    test('rank-deficient matrix zeroes small singular values', () {
      // Second column is a multiple of the first, rank 1
      final A = Matrix.fromList([
        [1, 2],
        [2, 4],
        [3, 6],
      ]);

      final AInverse = A.pseudoInverse(threshold: 1e-8);

      // Moore–Penrose
      final reconstructed = AInverse.multiply(A).multiply(AInverse);

      for (var i = 0; i < AInverse.rows; i++) {
        for (var j = 0; j < AInverse.cols; j++) {
          expect(
            (reconstructed.get(i, j) - AInverse.get(i, j)).abs() < 1e-8,
            isTrue,
          );
        }
      }
    });
  });
}
