// Ignored for json usage
// ignore_for_file: avoid_dynamic_calls

import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';
import 'package:test/test.dart';

import '../data/x02.dart';
import '../data/x42.dart';

void main() {
  group('Multivariate Linear Regression', () {
    test('works with 2 inputs and 3 outputs', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [4, 6, 5]);
      expect(p2, [8, 8, 8]);
    });

    test('works with 2 inputs and 3 outputs - intercept is 0', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [4, 6, 5]);
      expect(p2, [8, 8, 8]);
    });

    test('works with 2 inputs and 3 outputs - intercept is not 0', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [-1, 2, 10],
          [1, 6, 13],
          [3, 8, 15],
          [5, 10, 17],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [3, 8, 15]);
      expect(p2, [7, 10, 18]);
    });

    test('works with 2 inputs and 1 output (x02)', () {
      final x02X = x02Data['x']!;
      final x02Y = x02Data['y']!;

      final mlr = MultivariateLinearRegression(x: x02X, y: x02Y, intercept: false);
      final prediction = mlr.predictBatch(x02X);

      expect(prediction[0][0], closeTo(38.05, 0.01));
    });

    test('works with 2 inputs and 1 output (x42)', () {
      final x42X = x42Data['x']!;
      final x42Y = x42Data['y']!;

      final expectedWeights = [
        83.125,
        2.625,
        3.125,
        3.75,
        -2.0,
        -4.375,
        0.0,
        1.5,
        -0.25,
      ];

      final mlr = MultivariateLinearRegression(x: x42X, y: x42Y, intercept: false);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      for (var i = 0; i < weights.length; i++) {
        expect(weights[i][0], closeTo(expectedWeights[i], 1e-6));
      }
    });

    test('toJson and load', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      // final json = mlr.toJson();
      final loaded = MultivariateLinearRegression.load(mlr);

      final p = loaded.predict([2, 3]).map((e) => e.round()).toList();
      expect(p, [4, 6, 5]);
    });

    test('data mining test 1-1', () {
      final X = [
        [4.47],
        [208.3],
        [3400.0],
      ];

      final Y = [
        [0.51],
        [105.66],
        [1800.0],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(0.53, 0.01));
      expect(weights[1][0], closeTo(-3.29, 0.01));
    });

    test('data mining test 1-2', () {
      final X = <List<double>>[
        [4.47, 1],
        [208.3, 1],
        [3400.0, 1],
      ];

      final Y = <List<double>>[
        [0.51],
        [105.66],
        [1800.0],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y, intercept: false);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(0.53, 0.01));
      expect(weights[1][0], closeTo(-3.29, 0.01));
    });

    test('data mining test 2', () {
      final X = <List<double>>[
        [1, 1, 1],
        [2, 1, 1],
        [3, 1, 1],
      ];

      final Y = <List<double>>[
        [2, 3],
        [4, 6],
        [6, 9],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(2, 0.01));
      expect(weights[0][1], closeTo(3, 0.01));
      expect(weights[1][0], closeTo(0, 0.01));
      expect(weights[1][1], closeTo(0, 0.01));
      expect(weights[2][0], closeTo(0, 0.01));
      expect(weights[2][1], closeTo(0, 0.01));
    });

    test('data mining statistics test', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final json = mlr.toJson();
      final summary = json['summary'];

      expect(summary!['regressionStatistics']['standardError'], closeTo(6.27, 0.1));

      final vars = summary['variables'] as List;
      expect(vars[0]['coefficients'][0], closeTo(0.75, 0.01));
      expect(vars[0]['standardError'], closeTo(1.4, 0.01));
      expect(vars[0]['tStat'], closeTo(0.53, 0.01));

      expect(vars[1]['coefficients'][0], closeTo(5.25, 0.01));
      expect(vars[1]['standardError'], closeTo(2.43, 0.01));
      expect(vars[1]['tStat'], closeTo(2.16, 0.01));
      expect(vars[2]['label'], 'Intercept');
      expect(vars[2]['coefficients'][0], closeTo(13.75, 0.01));
      expect(vars[2]['standardError'], closeTo(7.81, 0.01));
      expect(vars[2]['tStat'], closeTo(1.76, 0.01));
    });

    test('statistics can be disabled', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y, statistics: false);
      final json = mlr.toJson();

      expect(json['summary'], isNull);
    });

    test('throws on wrong data type', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);

      expect(() => mlr.predict([3]), throwsArgumentError);
    });

    test('MLR typedef aliases MultivariateLinearRegression', () {
      final mlr = MLR(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      final prediction = mlr.predict([2, 3]);

      expect(mlr, isA<MultivariateLinearRegression>());

      expect(
        prediction.map((e) => e.round()).toList(),
        [4, 6, 5],
      );
    });

    test('forces SVD edge-case execution paths', () {
      final X = [
        [1e-10, 1e-10],
        [1e-10, 2e-10],
        [2e-10, 3e-10],
      ];

      final Y = [
        [1.0],
        [2.0],
        [3.0],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);

      final result = mlr.predict([1e-10, 2e-10]);

      expect(result.first.isFinite, true);
    });

    test('can fit model when x and y are not provided in constructor', () {
      final mlr = MLR()
        ..fit(
          x: [
            [0, 0],
            [1, 2],
            [2, 3],
            [3, 4],
          ],
          y: [
            [0, 0, 0],
            [2, 4, 3],
            [4, 6, 5],
            [6, 8, 7],
          ],
        );

      final p = mlr.predict([2, 3]).map((e) => e.round()).toList();
      expect(p, [4, 6, 5]);
    });
  });

  test('throws when fitting with mismatched x and y lengths', () {
    final mlr = MLR();

    expect(
      () => mlr.fit(
        x: [
          [0, 0],
          [1, 2],
        ],
        y: [
          [0, 0, 0],
        ],
      ),
      throwsArgumentError,
    );
  });

  test('throws when fitting with jagged x', () {
    final mlr = MLR();

    expect(
      () => mlr.fit(
        x: [
          [0, 0],
          [1, 2, 3],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
        ],
      ),
      throwsArgumentError,
    );
  });

  test('throws when fitting with jagged y', () {
    final mlr = MLR();

    expect(
      () => mlr.fit(
        x: [
          [0, 0],
          [1, 2],
        ],
        y: [
          [0, 0, 0],
          [2, 4],
        ],
      ),
      throwsArgumentError,
    );
  });

  test('throws when predicting with wrong input size', () {
    final mlr = MLR(
      x: [
        [0, 0],
        [1, 2],
        [2, 3],
        [3, 4],
      ],
      y: [
        [0, 0, 0],
        [2, 4, 3],
        [4, 6, 5],
        [6, 8, 7],
      ],
    );

    expect(() => mlr.predict([1]), throwsArgumentError);
    expect(() => mlr.predict([1, 2, 3]), throwsArgumentError);
  });

  test('throws when predict is called before fitting', () {
    final mlr = MLR();

    expect(() => mlr.predict([1, 2]), throwsArgumentError);
  });

  test('predictBatch returns predictions for all rows', () {
    final mlr = MLR(
      x: [
        [0, 0],
        [1, 2],
        [2, 3],
        [3, 4],
      ],
      y: [
        [0],
        [2],
        [4],
        [6],
      ],
    );

    final predictions = mlr.predictBatch([
      [2, 3],
      [3, 4],
    ]);

    expect(predictions.length, 2);
    expect(predictions[0][0], closeTo(4, 0.01));
    expect(predictions[1][0], closeTo(6, 0.01));
  });

  test('stdError is null when statistics are disabled', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
      statistics: false,
    );

    expect(mlr.stdError, isNull);
  });

  test('stdErrorMatrix throws when statistics are disabled', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
      statistics: false,
    );

    expect(
      () => mlr.stdErrorMatrix,
      throwsStateError,
    );
  });

  test('stdErrors returns one value per coefficient', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
        [4],
      ],
      y: [
        [2],
        [4],
        [6],
        [8],
      ],
    );

    expect(mlr.stdErrors.length, mlr.weights.length);
  });

  test('tStats returns one value per coefficient', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
        [4],
      ],
      y: [
        [2],
        [4],
        [6],
        [8],
      ],
    );

    expect(mlr.tStats.length, mlr.weights.length);
  });

  test('weights are empty before fitting', () {
    final mlr = MLR();

    expect(mlr.weights, isEmpty);
  });

  test('inputs and outputs are zero before fitting', () {
    final mlr = MLR();

    expect(mlr.inputs, 0);
    expect(mlr.outputs, 0);
  });

  test('load preserves configuration flags', () {
    final original = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
      intercept: false,
      statistics: false,
    );

    final loaded = MLR.load(original);

    expect(loaded.intercept, false);
    expect(loaded.statistics, false);
  });

  test('toJson exposes expected top level fields', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
    );

    final json = mlr.toJson();

    expect(json['name'], 'multivariateLinearRegression');
    expect(json['weights'], isNotNull);
    expect(json['inputs'], 1);
    expect(json['outputs'], 1);
    expect(json['intercept'], true);
  });

  test('predict works for model without intercept', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
      intercept: false,
    );

    final prediction = mlr.predict([5]);

    expect(prediction[0], closeTo(10, 0.01));
  });

  test('refit replaces previously learned coefficients', () {
    final mlr = MLR(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [2],
        [4],
        [6],
      ],
    );

    final firstPrediction = mlr.predict([5])[0];

    mlr.fit(
      x: [
        [1],
        [2],
        [3],
      ],
      y: [
        [3],
        [6],
        [9],
      ],
    );

    final secondPrediction = mlr.predict([5])[0];

    expect(firstPrediction, isNot(closeTo(secondPrediction, 0.01)));
    expect(secondPrediction, closeTo(15, 0.01));
  });

  test('stdError is computed when statistics are enabled', () {
    final mlr = MLR(
      x: [
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ],
      y: [
        [19],
        [28],
        [37],
        [46],
        [40],
      ],
    );

    expect(mlr.stdError, isNotNull);
    expect(mlr.stdError! > 0, true);
  });
}
