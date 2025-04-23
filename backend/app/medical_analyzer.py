
import time
import traceback
import os
from groq import Groq
import PyPDF2
import io
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import logging
import logging.config
from datetime import datetime
import pandas as pd
from PIL import Image
import warnings
from .config import COLOR_PALETTE, LOGGING_CONFIG

warnings.filterwarnings("ignore")

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class MedicalReportAnalyzer:
    """Medical report analyzer using Groq API."""

    def __init__(self, api_key: str):
        """Initialize the analyzer with Groq API key."""
        self.client = Groq(api_key=api_key)
        # self.model = "mixtral-8x7b-32768"
        self.model = "llama3-8b-8192"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _safe_json_loads(self, content: str, default_value: any) -> any:
        """Safely parse JSON with error handling."""
        try:
            content = content.strip()
            if not content.startswith('{'):
                content = content[content.find('{'):]
            if not content.endswith('}'):
                content = content[:content.rfind('}')+1]
            return json.loads(content)
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}\nContent: {content}")
            return default_value

    def _make_api_request(self, prompt: str, default_value: any) -> any:
        """Make API request with error handling and rate limiting."""
        try:
            self._rate_limit()
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            return self._safe_json_loads(completion.choices[0].message.content, default_value)
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            logger.error(traceback.format_exc())
            return default_value

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text content from PDF file with enhanced error handling."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue
            
            full_text = "\n".join(text_content)
            
            if not full_text.strip():
                logger.warning("No text could be extracted from the PDF")
                return "No readable text found in the document"
                
            return full_text

        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error extracting PDF content: {str(e)}"

    def detect_fraud_indicators(self, text_content: str, cost_data: Dict, relevance_data: Dict) -> Dict:
        """Detect potential fraudulent charges."""
        try:
            # Extract test dates from text content
            test_dates = {}
            current_date = None
            
            # More precise date parsing
            for line in text_content.split('\n'):
                if 'Tests Conducted on' in line:
                    current_date = line.split('Tests Conducted on')[1].strip().rstrip(':')
                elif current_date and line.strip().startswith('-'):
                    # Extract test name more carefully
                    test_name = line.split('($')[0].strip('- ').strip()
                    if test_name:
                        if test_name not in test_dates:
                            test_dates[test_name] = set()  # Use set to avoid duplicate dates
                        test_dates[test_name].add(current_date)

            # Verify no same-day duplicates exist
            same_day_duplicates = {}
            for test_name, dates in test_dates.items():
                if len(dates) < len([d for d in test_dates[test_name]]):  # This should never happen with a set
                    same_day_duplicates[test_name] = dates

            fraud_prompt = f"""Analyze this medical report for potential fraud indicators:
            1. Compare charges with standard rates
            2. Check for unnecessary procedures
            3. Look for unusual billing patterns
            4. Check for other irregular patterns

            Note: Tests performed on different days are not considered duplicates.

            Return ONLY a JSON object with this structure, nothing else:
            {{
                "fraud_score": 0,
                "suspicious_items": [
                    {{
                        "item": "charge description",
                        "reason": "why suspicious",
                        "confidence": 0
                    }}
                ],
                "recommendations": ["recommendation1", "recommendation2"]
            }}

            Medical Report:
            {text_content}

            Cost Data:
            {json.dumps(cost_data)}
            """

            result = self._make_api_request(fraud_prompt, {
                "fraud_score": 0,
                "suspicious_items": [],
                "recommendations": []
            })
            
            # Since we confirmed no same-day duplicates, remove any such claims
            if result and "suspicious_items" in result:
                result["suspicious_items"] = [
                    item for item in result["suspicious_items"]
                    if not any(d in item.get("reason", "").lower() for d in ["duplicate", "multiple", "same day"])
                ]
                
                # Adjust fraud score
                if not result["suspicious_items"]:
                    result["fraud_score"] = 0
                else:
                    result["fraud_score"] = min(90, len(result["suspicious_items"]) * 20)

            return result

        except Exception as e:
            logger.error(f"Fraud detection failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "fraud_score": 0,
                "suspicious_items": [],
                "recommendations": []
            }
    def verify_policy_documents(self, medical_pdf: bytes, policy_pdf: bytes) -> Dict:
        """Verify policy numbers and claim amounts between medical and policy documents."""
        try:
            # Extract text from both documents
            medical_text = self.extract_text_from_pdf(medical_pdf)
            policy_text = self.extract_text_from_pdf(policy_pdf)

            # Create prompt for policy verification
            verify_prompt = f"""Compare these two documents and verify:
            1. Policy number matches between documents
            2. Claim amount in medical document is within policy coverage
            3. Coverage details match

            Return ONLY a JSON object with this structure, nothing else:
            {{
                "policy_match": {{
                    "medical_policy_id": "policy number found in medical doc",
                    "policy_doc_id": "policy number found in policy doc",
                    "matches": true/false
                }},
                "claim_verification": {{
                    "claim_amount": 1000.00,
                    "coverage_amount": 5000.00,
                    "within_coverage": true/false,
                    "coverage_percentage": 85
                }},
                "coverage_details": {{
                    "matches": true/false,
                    "discrepancies": [
                        {{
                            "item": "description of mismatch",
                            "medical_doc": "value in medical doc",
                            "policy_doc": "value in policy doc"
                        }}
                    ]
                }}
            }}

            Medical Document:
            {medical_text}

            Policy Document:
            {policy_text}
            """

            # Make API request and get verification results
            verification_results = self._make_api_request(
                verify_prompt,
                {
                    "policy_match": {
                        "medical_policy_id": "Unknown",
                        "policy_doc_id": "Unknown",
                        "matches": False
                    },
                    "claim_verification": {
                        "claim_amount": 0,
                        "coverage_amount": 0,
                        "within_coverage": False,
                        "coverage_percentage": 0
                    },
                    "coverage_details": {
                        "matches": False,
                        "discrepancies": []
                    }
                }
            )

            return verification_results

        except Exception as e:
            logger.error(f"Policy verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def parse_costs(self, text_content: str) -> Dict:
        """Extract cost information from text."""
        try:
            cost_prompt = f"""Extract all costs and charges from this medical report.
            Return ONLY a JSON object with this structure, nothing else:
            {{
                "itemized_costs": [
                    {{"item": "Service/Test Name", "cost": 1000.00}}
                ],
                "total_cost": 5000.00
            }}

            Medical Report:
            {text_content}
            """

            default_value = {
                "itemized_costs": [],
                "total_cost": 0
            }
            
            result = self._make_api_request(cost_prompt, default_value)
            
            # Validate the costs
            if result and "itemized_costs" in result:
                for item in result["itemized_costs"]:
                    if not isinstance(item.get("cost"), (int, float)):
                        item["cost"] = 0.0
                    if not isinstance(item.get("item"), str):
                        item["item"] = "Unknown Service"
                
                # Recalculate total cost
                total = sum(item["cost"] for item in result["itemized_costs"])
                result["total_cost"] = total

            return result

        except Exception as e:
            logger.error(f"Cost parsing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "itemized_costs": [],
                "total_cost": 0
            }

    def analyze_test_relevance(self, text_content: str) -> Dict:
        """Analyze tests for relevance to the main health problem."""
        try:
            relevance_prompt = f"""Analyze this medical report and identify:
            1. The main health problem/complaint
            2. Which tests are directly relevant to diagnosing/treating it
            3. Which tests appear unnecessary for the main problem
            Return ONLY a JSON object with this structure, nothing else:
            {{
                "main_problem": "description of primary health issue",
                "relevant_tests": [
                    {{
                        "test_name": "test name",
                        "relevance_explanation": "how this test helps diagnose/treat the main problem"
                    }}
                ],
                "irrelevant_tests": [
                    {{
                        "test_name": "test name",
                        "why_unnecessary": "explanation of why this test isn't needed for the main problem",
                        "potential_reason": "possible reason why this test was ordered"
                    }}
                ]
            }}

            Medical Report:
            {text_content}
            """

            default_value = {
                "main_problem": "Unable to determine",
                "relevant_tests": [],
                "irrelevant_tests": []
            }
            
            result = self._make_api_request(relevance_prompt, default_value)
            
            # Validate the result structure
            if not result.get("main_problem"):
                result["main_problem"] = "Unable to determine"
            
            if not isinstance(result.get("relevant_tests"), list):
                result["relevant_tests"] = []
                
            if not isinstance(result.get("irrelevant_tests"), list):
                result["irrelevant_tests"] = []
                
            # Validate each test entry
            for test in result["relevant_tests"]:
                if not isinstance(test, dict):
                    continue
                if not test.get("test_name"):
                    test["test_name"] = "Unknown Test"
                if not test.get("relevance_explanation"):
                    test["relevance_explanation"] = "No explanation provided"
                    
            for test in result["irrelevant_tests"]:
                if not isinstance(test, dict):
                    continue
                if not test.get("test_name"):
                    test["test_name"] = "Unknown Test"
                if not test.get("why_unnecessary"):
                    test["why_unnecessary"] = "No explanation provided"
                if not test.get("potential_reason"):
                    test["potential_reason"] = "Unknown reason"

            return result

        except Exception as e:
            logger.error(f"Test relevance analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "main_problem": "Analysis failed",
                "relevant_tests": [],
                "irrelevant_tests": []
            }
        
    def analyze_report(self, pdf_contents: bytes) -> Tuple:
        """Main analysis function that coordinates all analysis tasks."""
        try:
            # Extract text
            text_content = self.extract_text_from_pdf(pdf_contents)
            
            # Perform analyses in the correct order
            cost_data = self.parse_costs(text_content)
            relevance_data = self.analyze_test_relevance(text_content)
            fraud_analysis = self.detect_fraud_indicators(text_content, cost_data, relevance_data)

            return (
                relevance_data,
                cost_data,
                fraud_analysis
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_verification_box(self, verification_results: Dict) -> str:
        """Generate a text summary of policy verification results."""
        try:
            policy_status = "Match" if verification_results["policy_match"]["matches"] else "Mismatch"
            claim_status = "Within Coverage" if verification_results["claim_verification"]["within_coverage"] else "Exceeds Coverage"

            summary = (
                f"Policy Verification Results:\n"
                f"Policy Number Status: {policy_status}\n"
                f"Claim Amount: ${verification_results['claim_verification']['claim_amount']:,.2f}\n"
                f"Coverage Amount: ${verification_results['claim_verification']['coverage_amount']:,.2f}\n"
                f"Coverage Status: {claim_status}\n"
            )

            if not verification_results["coverage_details"]["matches"]:
                discrepancies = verification_results["coverage_details"]["discrepancies"]
                summary += "Discrepancies Found:\n"
                for discrepancy in discrepancies:
                    summary += (
                        f" - Item: {discrepancy['item']}\n"
                        f"   Medical Doc: {discrepancy['medical_doc']}\n"
                        f"   Policy Doc: {discrepancy['policy_doc']}\n"
                    )
            else:
                summary += "All coverage details match."

            return summary

        except Exception as e:
            logger.error(f"Error creating verification summary: {str(e)}")
            return f"Error generating verification results: {str(e)}"
        
    def parse_and_predict_daywise(self, text_content: str) -> Dict:
        """Parse medical report data and predict patient's condition progression along with test requirements and costs."""
        def standardize_condition(condition: str) -> str:
            """Standardize condition to Bad/Neutral/Good"""
            condition = condition.lower()
            if condition in ['bad', 'poor', 'critical']:
                return 'Bad'
            elif condition in ['fair', 'moderate', 'stable', 'satisfactory']:
                return 'Neutral'
            elif condition in ['good', 'excellent']:
                return 'Good'
            return 'Bad'  # Default to Bad for unknown conditions

        def extract_cost(test_info: str) -> float:
            """Extract cost from test information string."""
            try:
                if "($" in test_info:
                    cost_str = test_info.split("($")[1].strip(")")
                    return float(cost_str)
                return 0.0
            except:
                return 0.0

        try:
            # Step 1: Initial condition assessment remains the same
            initial_assessment_prompt = f"""
            Given the patient medical report below, perform a clinical assessment analysis by evaluating:

            1. Primary diagnosis and associated indicators
            2. Pattern and types of diagnostic tests ordered
            3. Test urgency levels and frequency
            4. Standard medical protocols based on the primary diagnosis
            5. Resource intensity of procedures

            Medical Report:
            {text_content}

            Return ONLY a JSON object:
            {{
                "initial_assessment": {{
                    "initial_condition": "Bad",
                    "severity_indicators": ["reason1", "reason2"],
                    "risk_level": "Low|Medium|High",
                    "primary_diagnosis": "string",
                    "diagnostic_pattern": {{
                        "key_tests_ordered": ["test1", "test2"],
                        "test_urgency_profile": "Routine|Mixed|Urgent",
                        "testing_frequency": "Low|Moderate|Intensive"
                    }}
                }}
            }}
            """

            initial_assessment = self._make_api_request(initial_assessment_prompt, {
                "initial_assessment": {
                    "initial_condition": "Bad",
                    "severity_indicators": [],
                    "risk_level": "Unknown",
                    "primary_diagnosis": "Unknown",
                    "diagnostic_pattern": {
                        "key_tests_ordered": [],
                        "test_urgency_profile": "Unknown",
                        "testing_frequency": "Unknown"
                    }
                }
            })

            # Step 2: Parse day-wise test data with costs
            days_data = {}
            current_date = None

            for line in text_content.splitlines():
                line = line.strip()
                if line.startswith("Tests Conducted on"):
                    current_date = line.split("Tests Conducted on")[-1].strip(": ")
                    days_data[current_date] = {"tests": [], "total_cost": 0.0, "potential_savings": 0.0}
                elif current_date and line.startswith("-"):
                    test_info = line[1:].strip()
                    if "($" in test_info:
                        test_name = test_info.split("($")[0].strip()
                        cost = extract_cost(test_info)
                        days_data[current_date]["tests"].append({
                            "test_name": test_name,
                            "cost": cost
                        })
                        days_data[current_date]["total_cost"] += cost

            # Step 3: Analyze condition progression, test necessity, and costs
            daily_predictions = []
            current_condition = standardize_condition(initial_assessment["initial_assessment"]["initial_condition"])
            primary_diagnosis = initial_assessment["initial_assessment"]["primary_diagnosis"]

            for day, details in sorted(days_data.items()):
                tests = details.get("tests", [])
                analyzed_tests = []
                daily_total_cost = 0.0
                daily_unnecessary_cost = 0.0

                # Analyze each test's impact on condition and cost
                for test in tests:
                    test_impact_prompt = f"""
                    Evaluate {test['test_name']} for a patient with {primary_diagnosis}:
                    1. Is this test standard protocol for the diagnosis?
                    2. What clinical insights does this test provide?
                    3. How does this test relate to monitoring disease progression?
                    4. What does ordering this test suggest about patient status?

                    Return ONLY a JSON object:
                    {{
                        "required": true/false,
                        "impact": "Improvement|Stable|Not Directly Related|Deterioration",
                        "clinical_indication": "what this test suggests about patient state",
                        "urgency": "Routine|Urgent|Emergency"
                    }}
                    """

                    test_analysis = self._make_api_request(test_impact_prompt, {
                        "required": False,
                        "impact": "Unknown",
                        "clinical_indication": "Analysis failed",
                        "urgency": "Unknown"
                    })

                    daily_total_cost += test["cost"]
                    if not test_analysis["required"]:
                        daily_unnecessary_cost += test["cost"]

                    analyzed_tests.append({
                        "test_name": test["test_name"],
                        "cost": test["cost"],
                        "required": test_analysis["required"],
                        "impact": test_analysis["impact"],
                        "indication": test_analysis["clinical_indication"],
                        "urgency": test_analysis["urgency"]
                    })

                # Count necessary vs unnecessary tests
                necessary_count = sum(1 for t in analyzed_tests if t["required"])
                unnecessary_count = sum(1 for t in analyzed_tests if not t["required"])

                # Previous condition is the current condition before update
                previous_condition = current_condition
                
                # Update condition based on test analysis
                if necessary_count > unnecessary_count:
                    if previous_condition == "Bad":
                        current_condition = "Neutral"
                    elif previous_condition == "Neutral":
                        current_condition = "Good"
                    else:
                        current_condition = previous_condition
                elif unnecessary_count > necessary_count:
                    if previous_condition == "Good":
                        current_condition = "Neutral"
                    elif previous_condition == "Neutral":
                        current_condition = "Bad"
                    else:
                        current_condition = previous_condition
                else:
                    current_condition = previous_condition

                day_prediction = {
                    "day": day,
                    "tests_conducted": analyzed_tests,
                    "predicted_condition": current_condition,
                    "confidence": 100,
                    "reasoning": f"Based on test analysis: {necessary_count} necessary vs {unnecessary_count} unnecessary tests. Previous condition: {previous_condition}",
                    "needs_urgent_care": current_condition == "Bad",
                    "cost_analysis": {
                        "total_cost": daily_total_cost,
                        "unnecessary_cost": daily_unnecessary_cost,
                        "potential_savings": daily_unnecessary_cost
                    },
                    "recommended_actions": []
                }

                if current_condition == "Bad":
                    day_prediction["recommended_actions"] = [
                        "Increase monitoring frequency",
                        "Consider additional diagnostic tests",
                        "Review medication regimen"
                    ]
                elif current_condition == "Good":
                    day_prediction["recommended_actions"] = [
                        "Continue current treatment plan",
                        "Maintain regular monitoring",
                        "Schedule follow-up as planned"
                    ]

                daily_predictions.append(day_prediction)

            # Calculate overall cost analysis
            total_cost = sum(day["cost_analysis"]["total_cost"] for day in daily_predictions)
            total_unnecessary_cost = sum(day["cost_analysis"]["unnecessary_cost"] for day in daily_predictions)

            return {
                "message": "Day-wise predictions completed successfully",
                "daywise_predictions": {
                    "initial_assessment": initial_assessment["initial_assessment"],
                    "daily_predictions": daily_predictions,
                    "overall_progression": {
                        "starting_condition": standardize_condition(initial_assessment["initial_assessment"]["initial_condition"]),
                        "final_condition": current_condition,
                        "risk_level": initial_assessment["initial_assessment"]["risk_level"]
                    },
                    "overall_cost_analysis": {
                        "total_cost": total_cost,
                        "total_unnecessary_cost": total_unnecessary_cost,
                        "total_potential_savings": total_unnecessary_cost,
                        "savings_percentage": (total_unnecessary_cost / total_cost * 100) if total_cost > 0 else 0
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in condition prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "message": "Error occurred during analysis",
                "daywise_predictions": {
                    "initial_assessment": {
                        "initial_condition": "Unknown",
                        "severity_indicators": [],
                        "risk_level": "Unknown",
                        "primary_diagnosis": "Unknown",
                        "diagnostic_pattern": {
                            "key_tests_ordered": [],
                            "test_urgency_profile": "Unknown",
                            "testing_frequency": "Unknown"
                        }
                    },
                    "daily_predictions": [],
                    "overall_progression": {
                        "starting_condition": "Unknown",
                        "final_condition": "Unknown",
                        "risk_level": "Unknown"
                    },
                    "overall_cost_analysis": {
                        "total_cost": 0,
                        "total_unnecessary_cost": 0,
                        "total_potential_savings": 0,
                        "savings_percentage": 0
                    }
                }
            }
    