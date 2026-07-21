| Column | Type | Description |
|---|---|---|
| `i` | `UInt32` | The column coordinate of the matrix representing a face of the QFSBM. |
| `j` | `UInt32` | The row coordinate of the matrix representing a face of the QFSBM. |
| `face` | `String` | The face of the QCube can be any from ["posx", "posy", "posz", "negx", "negy", "negz"]. |
| `area` | `Float32` | The area of a given cell in the QFSBM matrix (meters^2). |
| `g_00800mm_spc_obj_0000n00000_v042 facet_id` | `UInt64` | The facet_id occupying a given cell in the QFSBM matrix for the g_00800mm_spc_obj_0000n00000_v042.obj mesh. This uses the convention from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/. |
| `g_01600mm_spc_obj_0000n00000_v042 facet_id` | `UInt64` | The facet_id occupying a given cell in the QFSBM matrix for the g_01600mm_spc_obj_0000n00000_v042.obj mesh. This uses the convention from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/. |
| `g_03170mm_spc_obj_0000n00000_v020 facet_id` | `UInt64` | The facet_id occupying a given cell in the QFSBM matrix for the g_03170mm_spc_obj_0000n00000_v020.obj mesh. This uses the convention from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/. |
| `g_06310mm_spc_obj_0000n00000_v020 facet_id` | `UInt64` | The facet_id occupying a given cell in the QFSBM matrix for the g_06310mm_spc_obj_0000n00000_v020.obj mesh. This uses the convention from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/. |
| `TIR detailed_survey band depth 350` | `Float64` | The spices of interest defined as "BD350: Band depth at ~350 cm-1, defined as the average emissivity of OTES channels [34:35] plus the average emissivity of OTES channels [44:45] divided by two, divided by the average emissivity in channels [40:41].  NOTE: This value can be strongly affected by the mean emission angle of the observation - use with great caution, especially at high latitudes." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the detailed_survey phase of the mission. |
| `TIR detailed_survey band depth 440` | `Float64` | The spices of interest defined as "BD440: Band depth at ~440 cm-1, defined as the average emissivity of OTES channels [60:62] plus the average emissivity of OTES channels [43:45] divided by two, divided by the average emissivity in channels [50:51]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the detailed_survey phase of the mission. |
| `TIR detailed_survey slope 1000` | `Float64` | The spices of interest defined as "Slope1000: The "1000-800 cm-1" slope, defined as the average emissivity of OTES (level 3 x-axis) channels [113:115] divided by the average of channels [93:95]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the detailed_survey phase of the mission. |
| `TIR detailed_survey ratio 1000` | `Float64` | EMPTY COLUMN: NO DATA HERE see https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ |
| `TIR detailed_survey sigma band depth 350` | `Float64` | The uncertainty of the spices of interest in the column `TIR detailed_survey band depth 350`. |
| `TIR detailed_survey sigma band depth 440` | `Float64` | The uncertainty of the spices of interest in the column `TIR detailed_survey band depth 440`. |
| `TIR detailed_survey sigma slope 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR detailed_survey slope 1000`. |
| `TIR detailed_survey sigma ratio 1000` | `Float64` | EMPTY COLUMN: NO DATA HERE see https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ |
| `TIR detailed_survey band depth 350 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey band depth 350` columns uses. |
| `TIR detailed_survey band depth 440 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey band depth 440` columns uses. |
| `TIR detailed_survey slope 1000 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey slope 1000` columns uses. |
| `TIR detailed_survey ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey ratio 1000` columns uses. |
| `TIR detailed_survey sigma band depth 350 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey sigma band depth 350` columns uses. |
| `TIR detailed_survey sigma band depth 440 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey sigma band depth 440` columns uses. |
| `TIR detailed_survey sigma slope 1000 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey sigma slope 1000` columns uses. |
| `TIR detailed_survey sigma ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR detailed_survey sigma ratio 1000` columns uses. |
| `TIR recona band depth 350` | `Float64` | The spices of interest defined as "BD350: Band depth at ~350 cm-1, defined as the average emissivity of OTES channels [34:35] plus the average emissivity of OTES channels [44:45] divided by two, divided by the average emissivity in channels [40:41].  NOTE: This value can be strongly affected by the mean emission angle of the observation - use with great caution, especially at high latitudes." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the recona phase of the mission. |
| `TIR recona band depth 440` | `Float64` | The spices of interest defined as "BD440: Band depth at ~440 cm-1, defined as the average emissivity of OTES channels [60:62] plus the average emissivity of OTES channels [43:45] divided by two, divided by the average emissivity in channels [50:51]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the recona phase of the mission. |
| `TIR recona slope 1000` | `Float64` | EMPTY COLUMN: NO DATA HERE see https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ |
| `TIR recona ratio 1000` | `Float64` | The spices of interest defined as "Ratio1000: The "1000-800 cm-1" slope, defined as the average emissivity of OTES (level 3 x-axis) channels [113:115] divided by the average of channels [93:95]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the recona phase of the mission. |
| `TIR recona sigma band depth 350` | `Float64` | The uncertainty of the spices of interest in the column `TIR recona band depth 350`. |
| `TIR recona sigma band depth 440` | `Float64` | The uncertainty of the spices of interest in the column `TIR recona band depth 440`. |
| `TIR recona sigma slope 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR recona slope 1000`. |
| `TIR recona sigma ratio 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR recona ratio 1000`. |
| `TIR recona band depth 350 facet mesh` | `String` | The facet mesh that the `TIR recona band depth 350` columns uses. |
| `TIR recona band depth 440 facet mesh` | `String` | The facet mesh that the `TIR recona band depth 440` columns uses. |
| `TIR recona slope 1000 facet mesh` | `String` | The facet mesh that the `TIR recona slope 1000` columns uses. |
| `TIR recona ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR recona ratio 1000` columns uses. |
| `TIR recona sigma band depth 350 facet mesh` | `String` | The facet mesh that the `TIR recona sigma band depth 350` columns uses. |
| `TIR recona sigma band depth 440 facet mesh` | `String` | The facet mesh that the `TIR recona sigma band depth 440` columns uses. |
| `TIR recona sigma slope 1000 facet mesh` | `String` | The facet mesh that the `TIR recona sigma slope 1000` columns uses. |
| `TIR recona sigma ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR recona sigma ratio 1000` columns uses. |
| `TIR reconb band depth 350` | `Float64` | The spices of interest defined as "BD350: Band depth at ~350 cm-1, defined as the average emissivity of OTES channels [34:35] plus the average emissivity of OTES channels [44:45] divided by two, divided by the average emissivity in channels [40:41].  NOTE: This value can be strongly affected by the mean emission angle of the observation - use with great caution, especially at high latitudes." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconb phase of the mission. |
| `TIR reconb band depth 440` | `Float64` | The spices of interest defined as "BD440: Band depth at ~440 cm-1, defined as the average emissivity of OTES channels [60:62] plus the average emissivity of OTES channels [43:45] divided by two, divided by the average emissivity in channels [50:51]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconb phase of the mission. |
| `TIR reconb slope 1000` | `Float64` | EMPTY COLUMN: NO DATA HERE see https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ |
| `TIR reconb ratio 1000` | `Float64` | The spices of interest defined as "Ratio1000: The "1000-800 cm-1" slope, defined as the average emissivity of OTES (level 3 x-axis) channels [113:115] divided by the average of channels [93:95]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconb phase of the mission. |
| `TIR reconb sigma band depth 350` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconb band depth 350`. |
| `TIR reconb sigma band depth 440` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconb band depth 440`. |
| `TIR reconb sigma slope 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconb slope 1000`. |
| `TIR reconb sigma ratio 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconb ratio 1000`. |
| `TIR reconb band depth 350 facet mesh` | `String` | The facet mesh that the `TIR reconb band depth 350` columns uses. |
| `TIR reconb band depth 440 facet mesh` | `String` | The facet mesh that the `TIR reconb band depth 440` columns uses. |
| `TIR reconb slope 1000 facet mesh` | `String` | The facet mesh that the `TIR reconb slope 1000` columns uses. |
| `TIR reconb ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR reconb ratio 1000` columns uses. |
| `TIR reconb sigma band depth 350 facet mesh` | `String` | The facet mesh that the `TIR reconb sigma band depth 350` columns uses. |
| `TIR reconb sigma band depth 440 facet mesh` | `String` | The facet mesh that the `TIR reconb sigma band depth 440` columns uses. |
| `TIR reconb sigma slope 1000 facet mesh` | `String` | The facet mesh that the `TIR reconb sigma slope 1000` columns uses. |
| `TIR reconb sigma ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR reconb sigma ratio 1000` columns uses. |
| `TIR reconc band depth 350` | `Float64` | The spices of interest defined as "BD350: Band depth at ~350 cm-1, defined as the average emissivity of OTES channels [34:35] plus the average emissivity of OTES channels [44:45] divided by two, divided by the average emissivity in channels [40:41].  NOTE: This value can be strongly affected by the mean emission angle of the observation - use with great caution, especially at high latitudes." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconc phase of the mission. |
| `TIR reconc band depth 440` | `Float64` | The spices of interest defined as "BD440: Band depth at ~440 cm-1, defined as the average emissivity of OTES channels [60:62] plus the average emissivity of OTES channels [43:45] divided by two, divided by the average emissivity in channels [50:51]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconc phase of the mission. |
| `TIR reconc slope 1000` | `Float64` | EMPTY COLUMN: NO DATA HERE see https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ |
| `TIR reconc ratio 1000` | `Float64` | The spices of interest defined as "Ratio1000: The "1000-800 cm-1" slope, defined as the average emissivity of OTES (level 3 x-axis) channels [113:115] divided by the average of channels [93:95]." from https://sbnarchive.psi.edu/pds4/orex/orex.spectral_analysis_v1_0/data_tir_maps/ for the reconc phase of the mission. |
| `TIR reconc sigma band depth 350` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconc band depth 350`. |
| `TIR reconc sigma band depth 440` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconc band depth 440`. |
| `TIR reconc sigma slope 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconc slope 1000`. |
| `TIR reconc sigma ratio 1000` | `Float64` | The uncertainty of the spices of interest in the column `TIR reconc ratio 1000`. |
| `TIR reconc band depth 350 facet mesh` | `String` | The facet mesh that the `TIR reconc band depth 350` columns uses. |
| `TIR reconc band depth 440 facet mesh` | `String` | The facet mesh that the `TIR reconc band depth 440` columns uses. |
| `TIR reconc slope 1000 facet mesh` | `String` | The facet mesh that the `TIR reconc slope 1000` columns uses. |
| `TIR reconc ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR reconc ratio 1000` columns uses. |
| `TIR reconc sigma band depth 350 facet mesh` | `String` | The facet mesh that the `TIR reconc sigma band depth 350` columns uses. |
| `TIR reconc sigma band depth 440 facet mesh` | `String` | The facet mesh that the `TIR reconc sigma band depth 440` columns uses. |
| `TIR reconc sigma slope 1000 facet mesh` | `String` | The facet mesh that the `TIR reconc sigma slope 1000` columns uses. |
| `TIR reconc sigma ratio 1000 facet mesh` | `String` | The facet mesh that the `TIR reconc sigma ratio 1000` columns uses. |
| `VNIR detailed_survey band depth` | `Float64` |  |
| `VNIR detailed_survey reflectance` | `Float64` |  |
| `VNIR detailed_survey slope1 poly` | `Float64` |  |
| `VNIR detailed_survey slope2 poly` | `Float64` |  |
| `VNIR detailed_survey sigma band depth` | `Float64` |  |
| `VNIR detailed_survey sigma reflectance` | `Float64` |  |
| `VNIR detailed_survey sigma slope1 poly` | `Float64` |  |
| `VNIR detailed_survey sigma slope2 poly` | `Float64` |  |
| `VNIR detailed_survey band depth facet mesh` | `String` |  |
| `VNIR detailed_survey reflectance facet mesh` | `String` |  |
| `VNIR detailed_survey slope1 poly facet mesh` | `String` |  |
| `VNIR detailed_survey slope2 poly facet mesh` | `String` |  |
| `VNIR detailed_survey sigma band depth facet mesh` | `String` |  |
| `VNIR detailed_survey sigma reflectance facet mesh` | `String` |  |
| `VNIR detailed_survey sigma slope1 poly facet mesh` | `String` |  |
| `VNIR detailed_survey sigma slope2 poly facet mesh` | `String` |  |
| `VNIR recona band depth` | `Float64` |  |
| `VNIR recona reflectance` | `Float64` |  |
| `VNIR recona slope1 poly` | `Float64` |  |
| `VNIR recona slope2 poly` | `Float64` |  |
| `VNIR recona sigma band depth` | `Float64` |  |
| `VNIR recona sigma reflectance` | `Float64` |  |
| `VNIR recona sigma slope1 poly` | `Float64` |  |
| `VNIR recona sigma slope2 poly` | `Float64` |  |
| `VNIR recona band depth facet mesh` | `String` |  |
| `VNIR recona reflectance facet mesh` | `String` |  |
| `VNIR recona slope1 poly facet mesh` | `String` |  |
| `VNIR recona slope2 poly facet mesh` | `String` |  |
| `VNIR recona sigma band depth facet mesh` | `String` |  |
| `VNIR recona sigma reflectance facet mesh` | `String` |  |
| `VNIR recona sigma slope1 poly facet mesh` | `String` |  |
| `VNIR recona sigma slope2 poly facet mesh` | `String` |  |
| `VNIR reconb band depth` | `Float64` |  |
| `VNIR reconb reflectance` | `Float64` |  |
| `VNIR reconb slope1 poly` | `Float64` |  |
| `VNIR reconb slope2 poly` | `Float64` |  |
| `VNIR reconb sigma band depth` | `Float64` |  |
| `VNIR reconb sigma reflectance` | `Float64` |  |
| `VNIR reconb sigma slope1 poly` | `Float64` |  |
| `VNIR reconb sigma slope2 poly` | `Float64` |  |
| `VNIR reconb band depth facet mesh` | `String` |  |
| `VNIR reconb reflectance facet mesh` | `String` |  |
| `VNIR reconb slope1 poly facet mesh` | `String` |  |
| `VNIR reconb slope2 poly facet mesh` | `String` |  |
| `VNIR reconb sigma band depth facet mesh` | `String` |  |
| `VNIR reconb sigma reflectance facet mesh` | `String` |  |
| `VNIR reconb sigma slope1 poly facet mesh` | `String` |  |
| `VNIR reconb sigma slope2 poly facet mesh` | `String` |  |
| `VNIR reconc band depth` | `Float64` |  |
| `VNIR reconc reflectance` | `Float64` |  |
| `VNIR reconc slope1 poly` | `Float64` |  |
| `VNIR reconc slope2 poly` | `Float64` |  |
| `VNIR reconc sigma band depth` | `Float64` |  |
| `VNIR reconc sigma reflectance` | `Float64` |  |
| `VNIR reconc sigma slope1 poly` | `Float64` |  |
| `VNIR reconc sigma slope2 poly` | `Float64` |  |
| `VNIR reconc band depth facet mesh` | `String` |  |
| `VNIR reconc reflectance facet mesh` | `String` |  |
| `VNIR reconc slope1 poly facet mesh` | `String` |  |
| `VNIR reconc slope2 poly facet mesh` | `String` |  |
| `VNIR reconc sigma band depth facet mesh` | `String` |  |
| `VNIR reconc sigma reflectance facet mesh` | `String` |  |
| `VNIR reconc sigma slope1 poly facet mesh` | `String` |  |
| `VNIR reconc sigma slope2 poly facet mesh` | `String` |  |
| `uint8_reflectance` | `UInt8` |  |
| `32bit_reflectance` | `Float32` |  |
| `positions_x` | `Float32` |  |
| `positions_y` | `Float32` |  |
| `positions_z` | `Float32` |  |
| `detection_lod_level` | `UInt8` |  |
| `detection_lod_code` | `String` |  |
| `boulder_id` | `UInt32` |  |
| `r32_min` | `Float32` |  |
| `r32_max` | `Float32` |  |
| `r32_range` | `Float32` |  |
| `r32_mean` | `Float32` |  |
| `r32_std` | `Float32` |  |
| `r32_variance` | `Float32` |  |
| `r32_q25` | `Float32` |  |
| `r32_q75` | `Float32` |  |
| `r32_q90` | `Float32` |  |
| `r32_q95` | `Float32` |  |
| `r32_sum` | `Float32` |  |
| `r32_product` | `Float32` |  |
| `Area` | `Float32` |  |
| `center_x` | `Float32` |  |
| `min_x` | `Float32` |  |
| `max_x` | `Float32` |  |
| `center_y` | `Float32` |  |
| `min_y` | `Float32` |  |
| `max_y` | `Float32` |  |
| `center_z` | `Float32` |  |
| `min_z` | `Float32` |  |
| `max_z` | `Float32` |  |
| `number_of_samples` | `UInt32` |  |
| `g_00800mm_spc_obj_0000n00000_v042 facet_id alpha` | `UInt64` |  |
| `g_01600mm_spc_obj_0000n00000_v042 facet_id alpha` | `UInt64` |  |
| `g_03170mm_spc_obj_0000n00000_v020 facet_id alpha` | `UInt64` |  |
| `g_06310mm_spc_obj_0000n00000_v020 facet_id alpha` | `UInt64` |  |
