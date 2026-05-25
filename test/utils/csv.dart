// ignore_for_file: public_member_api_docs

import 'dart:io';

import 'package:csv/csv.dart';

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
