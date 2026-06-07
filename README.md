# Multivariate Linear Regression

[![Style: very good analysis][very_good_analysis_badge]][very_good_analysis_link]
![Test Coverage](https://img.shields.io/badge/Test%20coverage-97%25-green)
[![Powered by Mason](https://img.shields.io/endpoint?url=https%3A%2F%2Ftinyurl.com%2Fmason-badge)][mason_link]
[![License: MIT][license_badge]][license_link]

Multivariate linear regression for Dart with support for multiple outputs and optional intercept, implemented using Golub-Reinsch Singular Value Decomposition.

> **Inspired by [ml-matrix](https://github.com/mljs/matrix) and [regression-multivariate-linear](https://github.com/mljs/regression-multivariate-linear) Node.js libraries.**

---

## Installation

**In order to start using Multivariate Linear Regression you must have the [Dart SDK][dart_install_link] installed on your machine.**

Install via `dart pub add`:

```sh
dart pub add multivariate_linear_regression
```

---

## Usage

```dart
import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';

void main() {
  final x = [
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
  ];

  final y = [
    [0.0, 0.0, 0.0],
    [2.0, 4.0, 3.0],
    [4.0, 6.0, 5.0],
    [6.0, 8.0, 7.0],
  ];

  final mlr = MultivariateLinearRegression(
    x: x,
    y: y,
  );

  // OR
  // final mlr1 = MultivariateLinearRegression()..fit(x, y);

  // OR
  // final mlr2 = MLR(
  //   x: x,
  //   y: y,
  // );

  // OR
  // final ml3 = MLR()..fit(x, y);

  print(mlr.predict([3.0, 3.0]));
  print(mlr.predictBatch([[1.0, 2.0], [2.0, 3.0]]));
  print(mlr.weights);
  print(mlr.stdError);
  print(mlr.stdErrors);
  print(mlr.tStats);
  print(mlr.toJson());
}
```

---

## API Overview

### Constructor

```dart
MultivariateLinearRegression({
  required List<List<double>> x,
  required List<List<double>> y,
  bool intercept = true,
  bool statistics = true,
})
```

Creates a multivariate linear regression model.

- `x` — Input feature matrix (rows = samples, columns = features)
- `y` — Output matrix (rows = samples, columns = targets)
- `intercept` — Includes a bias (intercept) term when set to `true`
- `statistics` — Enables computation of additional metrics (standard errors, t-stats, etc.)

---

### Load Existing Model

```dart
factory MultivariateLinearRegression.load(MultivariateLinearRegression model)
```

Reconstructs a trained model from previously trained model

---

### Prediction

```dart
List<double> predict(List<double> input)
```

Returns predicted outputs for a single input vector.

```dart
List<List<double>> predictBatch(List<List<double>> inputs)
```

Returns predictions for multiple input rows.

---

### Coefficients & Metrics

```dart
List<List<double>> get weights
```

Matrix of regression coefficients (includes intercept if enabled).

```dart
double get stdError
```

Overall standard error of the model.

```dart
List<List<double>> get stdErrors
```

Standard error for each coefficient.

```dart
List<List<double>> get tStats
```

T-statistics corresponding to each coefficient.

```dart
List<List<double>> get stdErrorMatrix
```

Covariance matrix of the coefficients.

> Available only when `statistics = true`

---

### Serialization

```dart
Map<String, dynamic> toJson()
```

Serializes the model into a JSON-compatible format, including statistics when enabled.

---

## Contributing

Contributions are welcome. Before opening a pull request, please read the contributing guide:

- [Contributing Guide][contributing_link]

---

## Support

If you find this package useful, please consider supporting it:

- Like the [package on pub.dev](https://pub.dev/packages/multivariate_linear_regression)
- Star the [GitHub repository](https://github.com/noahweasley/Multivariate-Linear-Regression)

Your support helps improve the project and keeps it actively maintained 😊

[dart_install_link]: https://dart.dev/get-dart
[license_badge]: https://img.shields.io/badge/license-MIT-blue.svg
[license_link]: https://opensource.org/licenses/MIT
[mason_link]: https://github.com/felangel/mason
[very_good_analysis_badge]: https://img.shields.io/badge/style-very_good_analysis-B22C89.svg
[very_good_analysis_link]: https://pub.dev/packages/very_good_analysis
[contributing_link]: CONTRIBUTING.md
