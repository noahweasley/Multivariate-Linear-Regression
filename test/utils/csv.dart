import 'dart:io';

import 'package:csv/csv.dart';

// Read CSV file and convert to a matrix
List<List<double>> readCsv(String path) {
  final content = File(path).readAsStringSync();

  final rows = const CsvToListConverter(
    shouldParseNumbers: true,
  ).convert(content);

  return rows.map(_convertRowToDouble).toList();
}

List<double> _convertRowToDouble(List<dynamic> row) {
  final result = <double>[];

  for (final value in row) {
    result.add((value as num).toDouble());
  }

  return result;
}
