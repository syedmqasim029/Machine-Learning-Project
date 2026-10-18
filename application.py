import os
import sys

# Ensure project root is in sys.path for AWS deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = float(request.form.get("reading_score", 0))
        writing_score = float(request.form.get("writing_score", 0))

        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        raw_pred = predict_pipeline.predict(pred_df)

        # 1. Clip score to valid exam range [0, 100]
        final_score = round(max(0.0, min(100.0, float(raw_pred[0]))), 1)

        # 2. Performance Grade Indicator
        if final_score >= 90:
            grade = "A+ (Exceptional)"
            grade_color = "success"
        elif final_score >= 80:
            grade = "A (Excellent)"
            grade_color = "success"
        elif final_score >= 70:
            grade = "B (Good Standing)"
            grade_color = "primary"
        elif final_score >= 60:
            grade = "C (Satisfactory)"
            grade_color = "warning"
        elif final_score >= 50:
            grade = "D (Needs Improvement)"
            grade_color = "warning"
        else:
            grade = "F (At Risk)"
            grade_color = "danger"

        return render_template(
            "home.html",
            results=final_score,
            grade=grade,
            grade_color=grade_color,
            prev_data={
                "gender": gender,
                "ethnicity": ethnicity,
                "parental_level_of_education": parental_level_of_education,
                "lunch": lunch,
                "test_preparation_course": test_preparation_course,
                "reading_score": reading_score,
                "writing_score": writing_score,
            },
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
