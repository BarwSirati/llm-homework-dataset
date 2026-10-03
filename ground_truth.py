"""Expected JSON for each of the three tables.

Authored next to make_docx.py so the table content and the expected extraction
stay in sync. The notebook imports these to score the model's output.

Convention: body cells that are merged in the source document are expanded to
repeated values, so every row is a flat, complete record. This is the same
convention the extraction prompt asks the model to follow.
"""

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
    "complex": {
        # Two-level header: parent -> children. Leaf names become row keys.
        "headers": {
            "Student Information": ["Student ID", "Full Name"],
            "Assessment": ["Midterm (30)", "Final (40)", "Group Work (20)"],
            "Total Score (100)": [],
        },
        "rows": [
            {"Student ID": "65010001", "Full Name": "John Carter",
             "Midterm (30)": "28", "Final (40)": "35",
             "Group Work (20)": "18", "Total Score (100)": "81"},
            {"Student ID": "65010002", "Full Name": "Emily Watson",
             "Midterm (30)": "25", "Final (40)": "30",
             "Group Work (20)": "20", "Total Score (100)": "75"},
            {"Student ID": "65010003", "Full Name": "Michael Chen",
             "Midterm (30)": "30", "Final (40)": "38",
             "Group Work (20)": "19", "Total Score (100)": "87"},
            {"Student ID": "65010004", "Full Name": "Sarah Johnson",
             "Midterm (30)": "22", "Final (40)": "28",
             "Group Work (20)": "15", "Total Score (100)": "65"},
            {"Student ID": "65010005", "Full Name": "David Miller",
             "Midterm (30)": "19", "Final (40)": "24",
             "Group Work (20)": "14", "Total Score (100)": "57"},
        ],
    },
    "very_complex": {
        # Three-level header; each quarter splits into Plan / Actual.
        "headers": {
            "Category": [],
            "Item": [],
            "Quarter 1": ["Plan", "Actual"],
            "Quarter 2": ["Plan", "Actual"],
        },
        # Vertically merged category labels are repeated on every row they span.
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
    import json

    for name, table in GROUND_TRUTH.items():
        print(f"{name}: {len(table['rows'])} rows, "
              f"{len(table['rows'][0])} fields per row")
    print("\nSample row (very_complex):")
    print(json.dumps(GROUND_TRUTH["very_complex"]["rows"][0], indent=2))
