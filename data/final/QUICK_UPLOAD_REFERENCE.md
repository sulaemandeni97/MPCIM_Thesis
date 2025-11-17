# 🚀 Quick Upload Reference Card

## 📊 Available Datasets

| File | Rows | QA Coverage | Use Case |
|------|------|-------------|----------|
| `sample_dataset_100.csv` | 100 | 100% | Quick testing, demos |
| `integrated_full_dataset.csv` | 712 | 99.7% | Full analysis, production |
| `UPLOAD_TEMPLATE.csv` | 3 | 100% | Structure reference |

---

## ✅ Required Columns (19)

```
1. employee_id_hash          (string, unique)
2. company_id                (integer)
3. tenure_years              (integer)
4. gender                    (string: O/L)
5. marital_status            (string: married/single)
6. is_permanent              (boolean: t/f)
7. performance_score         (float: 0-100)
8. performance_rating        (string: Excellent/Good/Average/Poor)
9. has_promotion             (integer: 0/1)
10. behavior_avg             (float: 0-100)
11. psychological_score      (float: 0-100)
12. drive_score              (float: 0-100)
13. mental_strength_score    (float: 0-100)
14. adaptability_score       (float: 0-100)
15. collaboration_score      (float: 0-100)
16. has_quick_assessment     (integer: 0/1)
17. holistic_score           (float: 0-100)
18. score_alignment          (float: 0-1)
19. leadership_potential     (float: 0-100)
```

---

## 🔍 Quick Validation Checklist

Before uploading, ensure:
- [ ] All 19 columns present
- [ ] No duplicate employee_id_hash
- [ ] Scores within 0-100 range
- [ ] has_promotion is 0 or 1
- [ ] has_quick_assessment is 0 or 1
- [ ] No missing critical values

---

## 📝 Quick Example Row

```csv
employee_id_hash,company_id,tenure_years,gender,marital_status,is_permanent,performance_score,performance_rating,has_promotion,behavior_avg,psychological_score,drive_score,mental_strength_score,adaptability_score,collaboration_score,has_quick_assessment,holistic_score,score_alignment,leadership_potential
emp001,101,5,O,married,t,85.5,Good,1,88.3,23.5,75.2,78.5,76.8,82.1,1,65.2,0.65,58.3
```

---

## 🚀 Upload Steps

1. **Prepare CSV** → Use template as reference
2. **Open Data Explorer** → Navigate to page
3. **Click "Browse files"** → In sidebar
4. **Select CSV** → Choose your file
5. **Wait for validation** → Automatic checks
6. **Use dataset** → If validation passes

---

## ⚠️ Common Errors

| Error | Solution |
|-------|----------|
| Missing columns | Add all 19 required columns |
| Duplicate IDs | Ensure unique employee_id_hash |
| Invalid scores | Keep scores within 0-100 |
| Wrong data types | Check boolean fields (t/f or 0/1) |

---

## 💡 Quick Tips

- ✅ Start with `sample_dataset_100.csv` for testing
- ✅ Use `UPLOAD_TEMPLATE.csv` as structure reference
- ✅ Set QA scores to 0 if no QA data available
- ✅ Check `DATASET_UPLOAD_GUIDE.md` for details

---

## 📞 Need Help?

See full documentation: `DATASET_UPLOAD_GUIDE.md`

---

**Quick Reference v1.0** | Updated: Nov 17, 2025
