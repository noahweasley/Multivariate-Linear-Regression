import 'dart:math';
import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/pseudo_inverse.dart';
import 'package:multivariate_linear_regression/src/svd/svd.dart';

/// {@template multivariate_linear_regression}
/// Author: Ebenmelu Ifechukwu (@noahweasley)
///
/// Multivariate linear regression using Golub–Reinsch Singular Value Decomposition (SVD).
///
/// Normal equation:
///   β = (XᵀX)⁻¹ XᵀY
///
/// Where:
/// - X: design matrix
/// - Y: target matrix
/// - β: coefficient matrix
/// {@endtemplate}
class MultivariateLinearRegression {
  /// Creates a multivariate linear regression model.
  ///
  /// [x] is the input matrix with shape `(rows × features)`.
  ///
  /// [y] is the output matrix with shape `(rows × outputs)`.
  ///
  /// Both [x] and [y] must be provided together. If either is
  /// specified, the other must also be specified. Alternatively,
  /// both may be omitted to create an untrained model.
  ///
  /// If [intercept] is `true`, a column of ones is added to the
  /// feature matrix before fitting.
  ///
  /// If [statistics] is `true`, additional regression statistics,
  /// such as variance estimates and inference metrics, are computed.
  ///
  /// When both [x] and [y] are provided, the model is fitted
  /// immediately by calling [fit].
  MultivariateLinearRegression({
    List<List<double>>? x,
    List<List<double>>? y,
    this.intercept = true,
    this.statistics = true,
  }) : assert(
          (x == null && y == null) || (x != null && y != null),
          'Both x and y must be provided together',
        ) {
    if (x != null && y != null) {
      fit(x: x, y: y);
    }
  }

  /// Recreates a model using the original training data.
  factory MultivariateLinearRegression.load(
    MultivariateLinearRegression model,
  ) {
    return MultivariateLinearRegression(
      x: model._originalX,
      y: model._originalY,
      intercept: model.intercept,
      statistics: model.statistics,
    );
  }

  /// Design matrix (X)
  late Matrix? _designMatrix;

  /// Design matrix (X) (Internal non-null accessors used after training).
  Matrix get _x => _designMatrix ?? (throw StateError('Model has not been fitted, call fit() with training data'));

  /// Output matrix (Y)
  late Matrix? _targetMatrix;

  /// Output matrix (Y) (Internal non-null accessors used after training).
  Matrix get _y => _targetMatrix ?? (throw StateError('Model has not been fitted, call fit() with training data'));

  /// Coefficient matrix (β)
  Matrix? _trainedBeta;

  /// Coefficient matrix (β) (Internal non-null accessors used after training).
  Matrix get _beta => _trainedBeta ?? (throw StateError('Model has not been fitted, call fit() with training data'));

  /// Number of input features (p)
  int? _inputs;

  /// Number of output variables (k)
  int? _outputs;

  /// Residual variance (σ²)
  double? _variance;

  /// Original input data (X)
  List<List<double>>? _originalX;

  /// Original output data (Y)
  late List<List<double>> _originalY;

  /// Whether an intercept column is included
  final bool intercept;

  /// Whether regression statistics are computed
  final bool statistics;

  /// Number of input features (p)
  int get inputs => _inputs ?? 0;

  /// Number of output variables (k)
  int get outputs => _outputs ?? 0;

  /// Regression coefficients (β)
  List<List<double>> get weights => _trainedBeta?.toList() ?? [];

  /// Standard error of the regression
  ///
  /// Formula:
  ///   σ = √σ²
  double? get stdError => _variance == null ? null : sqrt(_variance!);

  /// Fits the model to the provided training data.
  ///
  /// Can be used to fit a new model or refit an existing model with new data.
  ///
  /// [x] is the input matrix with shape `(rows × features)`.
  ///
  /// [y] is the output matrix with shape `(rows × outputs)`.
  void fit({required List<List<double>> x, required List<List<double>> y}) {
    _originalX = x;
    _originalY = y;

    _designMatrix = Matrix.fromList(x);
    _targetMatrix = Matrix.fromList(y);

    if (intercept) {
      final data = List.generate(_x.rows, (_) => [1.0]);
      final ones = Matrix.fromList(data);
      _designMatrix = _x.appendColumn(ones);
    }

    _inputs = _x.cols - (intercept ? 1 : 0);
    _outputs = _y.cols;

    _trainedBeta = _computeBeta();

    if (statistics) {
      _variance = _computeVariance();
    }
  }

  /// Computes the covariance matrix of the coefficients.
  ///
  /// Formula:
  ///   `Cov(β) = σ² · (XᵀX)⁻¹`
  ///
  /// Throws StateError if statistics are disabled.
  Matrix get stdErrorMatrix {
    if (_variance == null) {
      throw StateError('Statistics disabled');
    }

    final xtxInv = _x.transpose().multiply(_x).pseudoInverse();
    return xtxInv.scale(_variance!);
  }

  /// Computes standard errors for each coefficient.
  ///
  /// Formula:
  ///   `SE(βᵢ) = √Cov(βᵢ, βᵢ)`
  List<double> get stdErrors => stdErrorMatrix.diagonal().map(sqrt).toList();

  /// Computes t-statistics for each coefficient.
  ///
  /// Formula:
  ///  `t = β / SE(β)`
  List<double> get tStats {
    final errors = stdErrors;
    final coefficient = weights;

    return List.generate(coefficient.length, (i) {
      final beta = coefficient[i][0];
      return errors[i] == 0 ? 0.0 : beta / errors[i];
    });
  }

  /// Computes regression coefficients.
  ///
  /// Formula:
  ///   `β = (XᵀX)⁻¹ XᵀY`
  ///
  /// Where:
  /// - `X`is the feature matrix
  /// - `Y` is the target matrix
  /// - `β` is the coefficient matrix
  /// - `(XᵀX)⁻¹` is computed using SVD
  Matrix _computeBeta() {
    final xt = _x.transpose();
    final xx = xt.multiply(_x);
    final xy = xt.multiply(_y);

    final svdResults = GolubReinschSVD.decompose(xx).results;
    final invxx = svdResults.inverse();

    return xy.transpose().multiply(invxx).transpose();
  }

  /// Computes residual variance.
  ///
  /// Residuals:
  ///   r = Y − Xβ
  ///
  /// Variance:
  ///   σ² = Σ(rᵢ²) / (n − p)
  ///
  /// Where:
  /// - n = number of rows in Y
  /// - p = number of columns in X
  double _computeVariance() {
    final fitted = _x.multiply(_beta);
    final residuals = _y.add(fitted.neg());

    return residuals.toList().fold(0.0, (result, residual) => result + pow(residual[0], 2)) / (_y.rows - _x.cols);
  }

  /// Predicts output values for a single input vector.
  ///
  /// Throws ArgumentError if input size is incorrect.
  List<double> predict(List<double> x) {
    if (x.length != inputs) {
      throw ArgumentError(
        'Expected $inputs inputs, got ${x.length}',
      );
    }

    final result = List<double>.filled(outputs, 0.0);

    if (intercept) {
      for (var j = 0; j < outputs; j++) {
        result[j] = _beta.get(inputs, j);
      }
    }

    for (var i = 0; i < inputs; i++) {
      for (var j = 0; j < outputs; j++) {
        result[j] += _beta.get(i, j) * x[i];
      }
    }

    return result;
  }

  /// Predicts outputs for multiple input rows.
  List<List<double>> predictBatch(List<List<double>> x) => x.map(predict).toList();

  /// Converts the model to a JSON-compatible map.
  Map<String, dynamic> toJson() {
    return {
      'name': 'multivariateLinearRegression',
      'weights': weights,
      'inputs': inputs,
      'outputs': outputs,
      'intercept': intercept,
      'summary': statistics
          ? {
              'regressionStatistics': {
                'standardError': stdError,
              },
              'variables': List.generate(
                weights.length,
                (i) {
                  return {
                    'label': i == weights.length - 1 ? 'Intercept' : 'X${i + 1}',
                    'coefficients': weights[i],
                    'standardError': stdErrors[i],
                    'tStat': tStats[i],
                  };
                },
              ),
            }
          : null,
    };
  }
}
