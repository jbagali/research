#!/bin/bash

export ROOT_DIR=/home/grads/j/jbagali/researchGit/research
export MODULE_DIR=${ROOT_DIR}/tmp_output_files
export SCRIPTS_DIR=${ROOT_DIR}/scripts
export SYNTH_SCRIPT=${SCRIPTS_DIR}/yosys_simple_synth.tcl

#export SC_LIB=/home/abc586/tools/orfs/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
export SC_LIB=${SCRIPTS_DIR}/NangateOpenCellLibrary_typical.lib
#export ADDER_MAP_FILE=/home/abc586/tools/orfs/OpenROAD-flow-scripts/flow/platforms/nangate45/cells_adders.v
export ABC_DRIVER_CELL=BUF_X1
export ABC_LOAD_IN_FF=3.898

#export DESIGN_NICKNAME=gcd

#DESIGN_NAME and VERILOG_FILE will be changed to current tested module in LLMQueryEnv.py.
export DESIGN_NAME=counter
export VERILOG_FILE=counter.v
export OUTPUT_NAME=counter.v
export DUMP_DIR=${SCRIPTS_DIR}/dump/${OUTPUT_NAME%.v}
export ABC_AREA=1
#export ABC_CLOCK_PERIOD_IN_PS=460

#echo "SYNTH_SCRIPT: $SYNTH_SCRIPT"
#echo "DUMP_DIR: $DUMP_DIR"

yosys -c ${SYNTH_SCRIPT} > ${DUMP_DIR}/yosys_synth.log 2>&1
