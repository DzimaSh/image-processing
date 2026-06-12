# Vivado HLS Tcl Script for L2L Image Compression System

# Create a project
open_project -reset l2l_hls_project

# Add design files
add_files hardware/src/l2l_transform.cpp

# Add testbench files
add_files -tb hardware/src/l2l_tb.cpp
add_files -tb hardware/src/tb_vectors.txt

# Set top-level synthesizable function
set_top fwd_ladder_step_hls

# Create a solution
open_solution -reset "solution1"

# Target standard FPGA board part (Zynq-7000 xc7z020)
set_part {xc7z020clg400-1}

# Set clock period (10 ns = 100 MHz clock frequency)
create_clock -period 10 -name default

# 1. C-Simulation (Verifies logic correctness in C/C++ against golden vectors)
puts "=== Running HLS C-Simulation ==="
csim_design -clean

# 2. C-Synthesis (Compiles C/C++ to RTL, generates hardware architecture report)
puts "=== Running HLS C-Synthesis ==="
csynth_design

# 3. C/RTL Co-Simulation (Verifies synthesized Verilog/VHDL logic dynamically against C++ testbench)
puts "=== Running HLS C/RTL Co-Simulation ==="
cosim_design

# 4. Export IP Core (Generates IP Catalog ZIP for use in Vivado IP Integrator block designs)
puts "=== Exporting IP Core ==="
export_design -format ip_catalog -description "L2L Image Compression Forward Ladder Step IP Core" -vendor "BGUIR" -library "L2L"

exit
