import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Life Expectancy Predictor',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const PredictionPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  PredictionPageState createState() => PredictionPageState();
}

class PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // Controllers for ALL model features expected by the API (15 numeric + 1 categorical)
  final adultMortalityController = TextEditingController();
  final infantDeathsController = TextEditingController();
  final alcoholController = TextEditingController();
  final bmiController = TextEditingController();
  final hivAidsController = TextEditingController();
  final gdpController = TextEditingController();
  final schoolingController = TextEditingController();
  final healthcareIndexController = TextEditingController();
  final economicIndexController = TextEditingController();
  final womensEmpowermentController = TextEditingController();
  final nutritionIndexController = TextEditingController();
  final immunizationCoverageController = TextEditingController();
  final socioeconomicHealthController = TextEditingController();
  final developmentStageController = TextEditingController();

  // Dataset categorical feature
  final statusController = TextEditingController();

  String result = '';
  bool isLoading = false;

  // API endpoint - replace with your Render URL after deployment
  static const String apiUrl = 'http://127.0.0.1:8000';

  String? _validateRequiredNumberInRange(
    String? value,
    String fieldName,
    double min,
    double max,
  ) {
    final txt = value?.trim() ?? '';
    if (txt.isEmpty) return '$fieldName is required';
    final parsed = double.tryParse(txt);
    if (parsed == null) return 'Enter a valid number for $fieldName';
    if (parsed < min || parsed > max) return '$fieldName must be between $min and $max';
    return null;
  }

  String? _validateStatus(String? value) {
    final txt = value?.trim() ?? '';
    if (txt.isEmpty) return 'Status is required';
    final lower = txt.toLowerCase();
    if (lower == 'developing' || lower == 'developed') return null;
    return 'Status must be Developing or Developed';
  }

  double _parseRequiredInRange(
    TextEditingController controller,
    String fieldName,
    double min,
    double max,
  ) {
    final txt = controller.text.trim();
    if (txt.isEmpty) {
      throw Exception('Missing input: $fieldName');
    }
    final v = double.tryParse(txt);
    if (v == null) {
      throw Exception('Invalid number for $fieldName');
    }
    if (v < min || v > max) {
      throw Exception('Out of range for $fieldName. Expected $min to $max.');
    }
    return v;
  }

  Future<void> predict() async {
    if (isLoading) return;
    if (!(_formKey.currentState?.validate() ?? false)) return;
    
    setState(() {
      isLoading = true;
      result = '';
    });

    try {
      final rawStatus = statusController.text.trim();
      if (rawStatus.isEmpty) {
        throw Exception('Missing input: status (Developing or Developed)');
      }
      final statusLower = rawStatus.toLowerCase();
      final status = statusLower.contains('developed')
          ? 'Developed'
          : statusLower.contains('developing')
              ? 'Developing'
              : rawStatus;

      // Build request body with all required fields
      final requestBody = {
        'adult_mortality':
            _parseRequiredInRange(adultMortalityController, 'adult_mortality', 0, 1000),
        'infant_deaths':
            _parseRequiredInRange(infantDeathsController, 'infant_deaths', 0, 200),
        'alcohol': _parseRequiredInRange(alcoholController, 'alcohol', 0, 20),
        'bmi': _parseRequiredInRange(bmiController, 'bmi', 10, 50),
        'hiv_aids': _parseRequiredInRange(hivAidsController, 'hiv_aids', 0, 50),
        'gdp': _parseRequiredInRange(gdpController, 'gdp', 0, 5000000),
        'schooling': _parseRequiredInRange(schoolingController, 'schooling', 0, 25),
        'healthcare_index':
            _parseRequiredInRange(healthcareIndexController, 'healthcare_index', 0, 100),
        'economic_index': _parseRequiredInRange(economicIndexController, 'economic_index', 0, 1),
        'womens_empowerment':
            _parseRequiredInRange(womensEmpowermentController, 'womens_empowerment', 0, 1),
        'nutrition_index': _parseRequiredInRange(nutritionIndexController, 'nutrition_index', 0, 1),
        'immunization_coverage':
            _parseRequiredInRange(immunizationCoverageController, 'immunization_coverage', 0, 100),
        'socioeconomic_health':
            _parseRequiredInRange(socioeconomicHealthController, 'socioeconomic_health', 0, 1),
        'development_stage':
            _parseRequiredInRange(developmentStageController, 'development_stage', 0, 3),
        'status': status,
      };

      final response = await http.post(
        Uri.parse('$apiUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          result = '🎯 Predicted Life Expectancy: ${data['life_expectancy_years']?.toStringAsFixed(1) ?? 'N/A'} years\n'
                   'Model: ${data['model_used'] ?? 'Unknown'}';
        });
      } else if (response.statusCode == 422) {
        setState(() {
          result = '❌ Validation Error (datatype/range/missing fields):\n${response.body}';
        });
      } else {
        setState(() {
          result = '❌ Error: ${response.statusCode}\n${response.body}';
        });
      }
    } on SocketException {
      setState(() {
        result = '❌ Network error. Check internet connection and API availability.';
      });
    } catch (e) {
      setState(() {
        result = '❌ Input Error: $e';
      });
    } finally {
      setState(() => isLoading = false);
    }
  }

  Widget buildTextField({
    required TextEditingController controller,
    required String label,
    String? hint,
    double min = 0,
    double max = double.infinity,
    TextInputType? keyboardType,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: TextFormField(
        controller: controller,
        keyboardType: keyboardType ??
            TextInputType.numberWithOptions(decimal: true),
        autovalidateMode: AutovalidateMode.onUserInteraction,
        validator: (value) => _validateRequiredNumberInRange(value, label, min, max),
        decoration: InputDecoration(
          labelText: label,
          hintText: hint ?? '$min - $max',
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          prefixIcon: Icon(Icons.health_and_safety, size: 20),
        ),
      ),
    );
  }

  Widget buildStatusField() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: TextFormField(
        controller: statusController,
        keyboardType: TextInputType.text,
        autovalidateMode: AutovalidateMode.onUserInteraction,
        validator: _validateStatus,
        decoration: InputDecoration(
          labelText: 'Status (Developing/Developed)',
          hintText: 'Developing or Developed',
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          prefixIcon: Icon(Icons.flag, size: 20),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Life Expectancy Predictor'),
        backgroundColor: Colors.teal,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
              // Title Card
              Card(
                child: Padding(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Icon(Icons.monitor_heart, size: 64, color: Colors.teal),
                      SizedBox(height: 8),
                      Text(
                        'Enter country health data for prediction',
                        style: Theme.of(context).textTheme.titleMedium,
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
              SizedBox(height: 16),

              // Input Form - ALL 9 CORE FEATURES
              buildTextField(
                controller: adultMortalityController,
                label: 'Adult Mortality (per 1000)',
                hint: '0-1000',
              ),
              buildTextField(
                controller: infantDeathsController,
                label: 'Infant Deaths (per 1000)',
                hint: '0-200',
              ),
              buildTextField(
                controller: alcoholController,
                label: 'Alcohol Consumption (litres)',
                hint: '0-20',
              ),
              buildTextField(
                controller: bmiController,
                label: 'BMI Average',
                hint: '10-50',
              ),
              buildTextField(
                controller: hivAidsController,
                label: 'HIV/AIDS (%)',
                hint: '0-50',
              ),
              buildTextField(
                controller: gdpController,
                label: 'GDP per Capita',
                hint: '0+',
              ),
              buildTextField(
                controller: schoolingController,
                label: 'Schooling Years',
                hint: '0-25',
              ),
              buildTextField(
                controller: healthcareIndexController,
                label: 'Healthcare Index (0-100)',
                hint: '0-100',
              ),
              buildTextField(
                controller: economicIndexController,
                label: 'Economic Index (0-1)',
                hint: '0-1',
              ),

              buildTextField(
                controller: womensEmpowermentController,
                label: 'Womens Empowerment (0-1)',
                hint: '0-1',
              ),
              buildTextField(
                controller: nutritionIndexController,
                label: 'Nutrition Index (0-1)',
                hint: '0-1',
              ),
              buildTextField(
                controller: immunizationCoverageController,
                label: 'Immunization Coverage (0-100)',
                hint: '0-100',
              ),
              buildTextField(
                controller: socioeconomicHealthController,
                label: 'Socioeconomic Health (0-1)',
                hint: '0-1',
              ),
              buildTextField(
                controller: developmentStageController,
                label: 'Development Stage (0-3)',
                hint: '0-3',
                min: 0,
                max: 3,
              ),

              buildStatusField(),

              SizedBox(height: 24),

              // Predict Button
              ElevatedButton(
                onPressed: isLoading ? null : predict,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal,
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: isLoading
                    ? SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      )
                    : Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.psychology),
                          SizedBox(width: 8),
                          Text('Predict Life Expectancy', style: TextStyle(fontSize: 18)),
                        ],
                      ),
              ),

              SizedBox(height: 24),

              // Results Display
              if (result.isNotEmpty)
                Card(
                  color: result.contains('Error') || result.contains('❌') ? Colors.red.shade50 : Colors.green.shade50,
                  child: Padding(
                    padding: EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Icon(
                          result.contains('Error') || result.contains('❌') 
                              ? Icons.error : Icons.check_circle,
                          size: 48,
                          color: result.contains('Error') || result.contains('❌') 
                              ? Colors.red : Colors.green,
                        ),
                        SizedBox(height: 12),
                        Text(
                          result,
                          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),

              SizedBox(height: 32),

              // Info
              Text(
                'API: $apiUrl\nUpdate URL after Render deployment',
                style: TextStyle(fontSize: 12, color: Colors.grey, fontStyle: FontStyle.italic),
                textAlign: TextAlign.center,
              ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    // Clean up controllers
    adultMortalityController.dispose();
    infantDeathsController.dispose();
    alcoholController.dispose();
    bmiController.dispose();
    hivAidsController.dispose();
    gdpController.dispose();
    schoolingController.dispose();
    healthcareIndexController.dispose();
    economicIndexController.dispose();
    womensEmpowermentController.dispose();
    nutritionIndexController.dispose();
    immunizationCoverageController.dispose();
    socioeconomicHealthController.dispose();
    developmentStageController.dispose();
    statusController.dispose();
    super.dispose();
  }
}

