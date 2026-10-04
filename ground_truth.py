"""Expected JSON for each table — mirrors the copy embedded in the notebook.

Kept in sync so the standalone scripts and the notebook cannot drift apart.
"""

# the exact content of the three source tables.
GROUND_TRUTH = {
    "simple": {
        "headers": ["Student ID", "Full Name", "Department", "Grade"],
        "rows": [
            {"Student ID": "65010001", "Full Name": "John Carter",
             "Department": "Computer Engineering", "Grade": "A"},
            {"Student ID": "65010002", "Full Name": "Emily Watson",
             "Department": "Electrical Engineering", "Grade": "B+"},
            {"Student ID": "65010003", "Full Name": "Michael Chen",
             "Department": "Computer Engineering", "Grade": "A"},
            {"Student ID": "65010004", "Full Name": "Sarah Johnson",
             "Department": "Civil Engineering", "Grade": "B"},
            {"Student ID": "65010005", "Full Name": "David Miller",
             "Department": "Electrical Engineering", "Grade": "C+"},
        ],
    },
    # Header names follow prompt rule 2: parent_child for grouped columns.
    "complex": {
        "headers": ["Student Information_Student ID",
                    "Student Information_Full Name",
                    "Assessment_Midterm (30)", "Assessment_Final (40)",
                    "Assessment_Group Work (20)", "Total Score (100)"],
        "rows": [
            {"Student Information_Student ID": "65010001",
             "Student Information_Full Name": "John Carter",
             "Assessment_Midterm (30)": "28", "Assessment_Final (40)": "35",
             "Assessment_Group Work (20)": "18", "Total Score (100)": "81"},
            {"Student Information_Student ID": "65010002",
             "Student Information_Full Name": "Emily Watson",
             "Assessment_Midterm (30)": "25", "Assessment_Final (40)": "30",
             "Assessment_Group Work (20)": "20", "Total Score (100)": "75"},
            {"Student Information_Student ID": "65010003",
             "Student Information_Full Name": "Michael Chen",
             "Assessment_Midterm (30)": "30", "Assessment_Final (40)": "38",
             "Assessment_Group Work (20)": "19", "Total Score (100)": "87"},
            {"Student Information_Student ID": "65010004",
             "Student Information_Full Name": "Sarah Johnson",
             "Assessment_Midterm (30)": "22", "Assessment_Final (40)": "28",
             "Assessment_Group Work (20)": "15", "Total Score (100)": "65"},
            {"Student Information_Student ID": "65010005",
             "Student Information_Full Name": "David Miller",
             "Assessment_Midterm (30)": "19", "Assessment_Final (40)": "24",
             "Assessment_Group Work (20)": "14", "Total Score (100)": "57"},
        ],
    },
    "very_complex": {
        "headers": ["Category", "Item", "Quarter 1_Plan", "Quarter 1_Actual",
                    "Quarter 2_Plan", "Quarter 2_Actual"],
        "rows": [
            {"Category": "Personnel", "Item": "Salaries",
             "Quarter 1_Plan": "1,200", "Quarter 1_Actual": "1,180",
             "Quarter 2_Plan": "1,200", "Quarter 2_Actual": "1,195"},
            {"Category": "Personnel", "Item": "Overtime",
             "Quarter 1_Plan": "150", "Quarter 1_Actual": "162",
             "Quarter 2_Plan": "150", "Quarter 2_Actual": "148"},
            {"Category": "Personnel", "Item": "Benefits",
             "Quarter 1_Plan": "80", "Quarter 1_Actual": "78",
             "Quarter 2_Plan": "80", "Quarter 2_Actual": "85"},
            {"Category": "Personnel", "Item": "Subtotal",
             "Quarter 1_Plan": "1,430", "Quarter 1_Actual": "1,420",
             "Quarter 2_Plan": "1,430", "Quarter 2_Actual": "1,428"},
            {"Category": "Operations", "Item": "Utilities",
             "Quarter 1_Plan": "300", "Quarter 1_Actual": "315",
             "Quarter 2_Plan": "300", "Quarter 2_Actual": "298"},
            {"Category": "Operations", "Item": "Office Supplies",
             "Quarter 1_Plan": "120", "Quarter 1_Actual": "110",
             "Quarter 2_Plan": "120", "Quarter 2_Actual": "131"},
            {"Category": "Operations", "Item": "Subtotal",
             "Quarter 1_Plan": "420", "Quarter 1_Actual": "425",
             "Quarter 2_Plan": "420", "Quarter 2_Actual": "429"},
            {"Category": "Capital", "Item": "Computer Equipment",
             "Quarter 1_Plan": "900", "Quarter 1_Actual": "880",
             "Quarter 2_Plan": "450", "Quarter 2_Actual": "460"},
            {"Category": "Capital", "Item": "Subtotal",
             "Quarter 1_Plan": "900", "Quarter 1_Actual": "880",
             "Quarter 2_Plan": "450", "Quarter 2_Actual": "460"},
            {"Category": "Grand Total", "Item": "Grand Total",
             "Quarter 1_Plan": "2,750", "Quarter 1_Actual": "2,725",
             "Quarter 2_Plan": "2,300", "Quarter 2_Actual": "2,317"},
        ],
    },
}


if __name__ == "__main__":
    for name, gt in GROUND_TRUTH.items():
        print(f"{name:13s} {len(gt['headers'])} columns x {len(gt['rows'])} rows")
