set config_proj [lindex $argv 0]

open_project $config_proj
reset_run synth_1
launch_runs -to_step write_bitstream impl_1 -jobs 8
wait_on_run impl_1
close_project
