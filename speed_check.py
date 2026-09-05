import csv
import time
from custom_float_wp import CustomFloat

# Setup Custom Float (7 Exponent bits, 8 Mantissa bits -> FP16 variant)
cf = CustomFloat(10, 23)

log_filename = "operation_log_1M_4.csv"
summary_filename = "benchmark_summary_1M_4.csv"

N = 2_000_000
LOG_INTERVAL = 100

x_fp32 = 1.2345
x_custom = cf.quantize(1.2345)

total_abs_error = 0.0
total_rel_error = 0.0

max_abs_error = 0.0
min_abs_error = float("inf")

max_rel_error = 0.0
min_rel_error = float("inf")

start = time.perf_counter()

with open(log_filename, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Iteration",
            "FP32 Input",
            "Custom Input",
            "FP32 Result",
            "Custom Result",
            "Absolute Error",
            "Relative Error (%)",
        ]
    )

    for i in range(N):
        # -------- FP32 --------
        fp32_result = x_fp32 * 1.0001 + 0.0002

        # -------- Custom FP --------
        custom_result = cf.quantize(
            cf.quantize(x_custom * 1.0001) + cf.quantize(0.0002)
        )

        abs_error = abs(fp32_result - custom_result)
        rel_error = (
            (abs_error / abs(fp32_result) * 100) if fp32_result != 0 else 0.0
        )

        # Log periodically or at boundaries
        if (i + 1) % LOG_INTERVAL == 0 or i == 0 or i == N - 1:
            writer.writerow(
                [
                    i + 1,
                    x_fp32,
                    x_custom,
                    fp32_result,
                    custom_result,
                    abs_error,
                    rel_error,
                ]
            )

        # Update values for next iteration
        x_fp32 = fp32_result
        x_custom = custom_result

        # Accumulate metrics
        total_abs_error += abs_error
        total_rel_error += rel_error

        if abs_error > max_abs_error:
            max_abs_error = abs_error
        if abs_error < min_abs_error:
            min_abs_error = abs_error

        if rel_error > max_rel_error:
            max_rel_error = rel_error
        if rel_error < min_rel_error:
            min_rel_error = rel_error

end = time.perf_counter()
elapsed_time = end - start

# Write summary file
with open(summary_filename, "w", newline="") as summary_file:
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(["Metric", "Value"])
    summary_writer.writerow(["Iterations", N])
    summary_writer.writerow(
        ["Custom Format", f"E{cf.exponent_bits}M{cf.mantissa_bits}"]
    )
    summary_writer.writerow(["Initial Value", 1.2345])
    summary_writer.writerow(["Operation", "x = x * 1.0001 + 0.0002"])
    summary_writer.writerow(["Execution Time (sec)", elapsed_time])
    summary_writer.writerow(["Operations/sec", N / elapsed_time])
    summary_writer.writerow(["Final FP32 Value", x_fp32])
    summary_writer.writerow(["Final Custom Value", x_custom])
    summary_writer.writerow(["Final Absolute Error", abs(x_fp32 - x_custom)])
    summary_writer.writerow(
        [
            "Final Relative Error (%)",
            abs(x_fp32 - x_custom) / abs(x_fp32) * 100 if x_fp32 != 0 else 0,
        ]
    )
    summary_writer.writerow(["Average Absolute Error", total_abs_error / N])
    summary_writer.writerow(["Average Relative Error (%)", total_rel_error / N])
    summary_writer.writerow(["Maximum Absolute Error", max_abs_error])
    summary_writer.writerow(["Maximum Relative Error (%)", max_rel_error])
    summary_writer.writerow(["Minimum Absolute Error", min_abs_error])
    summary_writer.writerow(["Minimum Relative Error (%)", min_rel_error])

# Console Output
print("\n" + "=" * 70)
print("               CUSTOM FLOAT BENCHMARK COMPLETED")
print("=" * 70)
print(f"Custom Format        : FP{1 + cf.exponent_bits + cf.mantissa_bits}")
print(f"Exponent Bits        : {cf.exponent_bits}")
print(f"Mantissa Bits        : {cf.mantissa_bits}")
print(f"Iterations           : {N:,}\n")

print(f"Execution Time       : {elapsed_time:.6f} sec")
print(f"Operations / Second  : {N / elapsed_time:,.2f}\n")

print(f"Final FP32 Value     : {x_fp32}")
print(f"Final Custom Value   : {x_custom}\n")

print(f"Average Abs Error    : {total_abs_error / N:.10f}")
print(f"Average Rel Error    : {total_rel_error / N:.6f}%")
print(f"Maximum Abs Error    : {max_abs_error:.10f}")
print(f"Maximum Rel Error    : {max_rel_error:.6f}%\n")

print("Generated Files")
print("----------------")
print(f"✓ {log_filename}")
print(f"✓ {summary_filename}")
print("=" * 70)