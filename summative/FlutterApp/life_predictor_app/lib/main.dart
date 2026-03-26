import 'package:flutter/material.dart';
import 'dart:convert';
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
  static const String apiUrl = 'https://linear-regression-model-x6f4.onrender.com';

  double _parseRequiredDouble(TextEditingController controller, String fieldName) {
    final txt = controller.text.trim();
    if (txt.isEmpty) {
      throw Exception('Missing input: $fieldName');
    }
    final v = double.tryParse(txt);
    if (v == null) {
      throw Exception('Invalid number for $fieldName');
    }
    return v;
  }

  Future<void> predict() async {
    if (isLoading) return;
    
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
        'adult_mortality': _parseRequiredDouble(adultMortalityController, 'adult_mortality'),
        'infant_deaths': _parseRequiredDouble(infantDeathsController, 'infant_deaths'),
        'alcohol': _parseRequiredDouble(alcoholController, 'alcohol'),
        'bmi': _parseRequiredDouble(bmiController, 'bmi'),
        'hiv_aids': _parseRequiredDouble(hivAidsController, 'hiv_aids'),
        'gdp': _parseRequiredDouble(gdpController, 'gdp'),
        'schooling': _parseRequiredDouble(schoolingController, 'schooling'),
        'healthcare_index': _parseRequiredDouble(healthcareIndexController, 'healthcare_index'),
        'economic_index': _parseRequiredDouble(economicIndexController, 'economic_index'),
        'womens_empowerment':
            _parseRequiredDouble(womensEmpowermentController, 'womens_empowerment'),
        'nutrition_index': _parseRequiredDouble(nutritionIndexController, 'nutrition_index'),
        'immunization_coverage':
            _parseRequiredDouble(immunizationCoverageController, 'immunization_coverage'),
        'socioeconomic_health':
            _parseRequiredDouble(socioeconomicHealthController, 'socioeconomic_health'),
        'development_stage': _parseRequiredDouble(developmentStageController, 'development_stage'),
        'status': status,
      };

      final response = await http.post(
        Uri.parse(apiUrl),
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
      child: TextField(
        controller: controller,
        keyboardType: keyboardType ??
            TextInputType.numberWithOptions(decimal: true),
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
      child: TextField(
        controller: statusController,
        keyboardType: TextInputType.text,
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

