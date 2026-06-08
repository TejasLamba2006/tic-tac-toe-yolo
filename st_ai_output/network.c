/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-08T10:41:54+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x1201dc8e29f691176dd58b5552b54b78"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-06-08T10:41:54+0530"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  images_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 307200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  images_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 307200, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 409600, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 409600, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 409600, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 307200, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38400, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_m_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_m_1_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_m_2_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  _model_10_Resize_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  _model_11_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 153600, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 76800, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  _model_13_Resize_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 204800, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  _model_14_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 307200, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 153600, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4800, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  _model_17_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 76800, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 76800, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 51200, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1200, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  _model_20_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38400, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_Split_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_Split_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38400, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 300, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Concat_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6300, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6300, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 134400, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_Reshape_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 134400, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_Transpose_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 134400, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_Softmax_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 134400, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8400, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8400, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chfirst_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8400, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Slice_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Slice_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Add_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Sub_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Add_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Div_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Concat_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8400, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8400, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  output0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 14700, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  output0_Transpose_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 14700, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Constant_14_output_0_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Constant_15_output_0_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2100, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#257 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#258 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#259 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#260 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#261 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#262 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#263 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#264 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#265 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#266 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#267 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 73728, AI_STATIC)

/* Array#268 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#269 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#270 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#271 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#272 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#273 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#274 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#275 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#276 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#277 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#278 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#279 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#280 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#281 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#282 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 294912, AI_STATIC)

/* Array#283 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#284 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#285 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#286 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#287 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#288 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#289 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#290 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#291 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98304, AI_STATIC)

/* Array#292 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#293 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#294 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#295 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 131072, AI_STATIC)

/* Array#296 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#297 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#298 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#299 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#300 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#301 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#302 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#303 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#304 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#305 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#306 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#307 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#308 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#309 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#310 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#311 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#312 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#313 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#314 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#315 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#316 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#317 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#318 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#319 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#320 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#321 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#322 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#323 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#324 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#325 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#326 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#327 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#328 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#329 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#330 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#331 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#332 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#333 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#334 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#335 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#336 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#337 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#338 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 73728, AI_STATIC)

/* Array#339 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#340 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#341 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#342 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#343 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#344 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 73728, AI_STATIC)

/* Array#345 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#346 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#347 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#348 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#349 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#350 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#351 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#352 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98304, AI_STATIC)

/* Array#353 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#354 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_Split_output_0_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#355 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#356 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#357 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#358 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#359 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98304, AI_STATIC)

/* Array#360 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#361 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#362 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#363 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#364 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#365 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#366 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#367 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#368 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#369 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#370 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#371 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#372 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#373 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#374 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#375 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Constant_13_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#376 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_Constant_12_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4200, AI_STATIC)

/* Array#377 */
AI_ARRAY_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27, AI_STATIC)

/* Array#378 */
AI_ARRAY_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#379 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#380 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#381 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#382 */
AI_ARRAY_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#383 */
AI_ARRAY_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#384 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#385 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#386 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#387 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#388 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#389 */
AI_ARRAY_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#390 */
AI_ARRAY_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#391 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#392 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#393 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#394 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#395 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#396 */
AI_ARRAY_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#397 */
AI_ARRAY_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#398 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#399 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#400 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#401 */
AI_ARRAY_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#402 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#403 */
AI_ARRAY_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#404 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#405 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#406 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#407 */
AI_ARRAY_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#408 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#409 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#410 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#411 */
AI_ARRAY_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#412 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#413 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#414 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#415 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#416 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#417 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#418 */
AI_ARRAY_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#419 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#420 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#421 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#422 */
AI_ARRAY_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#423 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#424 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#425 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#426 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#427 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#428 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#429 */
AI_ARRAY_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#430 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#431 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#432 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#433 */
AI_ARRAY_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#434 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#435 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#436 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#437 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#438 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#439 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#440 */
AI_ARRAY_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_act_Mul_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 160, 160), AI_STRIDE_INIT(4, 4, 4, 64, 10240),
  1, &_model_0_act_Mul_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_act_Sigmoid_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 160, 160), AI_STRIDE_INIT(4, 4, 4, 64, 10240),
  1, &_model_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_model_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 160, 160), AI_STRIDE_INIT(4, 4, 4, 64, 10240),
  1, &_model_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 3), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &_model_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 3, 3, 3, 16), AI_STRIDE_INIT(4, 4, 12, 192, 576),
  1, &_model_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _model_10_Resize_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 20, 20), AI_STRIDE_INIT(4, 4, 4, 1024, 20480),
  1, &_model_10_Resize_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _model_11_Concat_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 20, 20), AI_STRIDE_INIT(4, 4, 4, 1536, 30720),
  1, &_model_11_Concat_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_Concat_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 20, 20), AI_STRIDE_INIT(4, 4, 4, 768, 15360),
  1, &_model_12_Concat_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_Split_output_0_num_or_size_splits, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_12_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_Split_output_0_output0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_Split_output_0_output0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_Split_output_0_output1, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_Split_output_0_output1_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_act_Mul_output_0_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_12_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &_model_12_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 384, 1, 1, 128), AI_STRIDE_INIT(4, 4, 1536, 196608, 196608),
  1, &_model_12_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_act_Mul_output_0_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_12_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_12_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_model_12_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 128), AI_STRIDE_INIT(4, 4, 768, 98304, 98304),
  1, &_model_12_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_12_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_12_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_12_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_12_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_12_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_12_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_12_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _model_13_Resize_output_0_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 40, 40), AI_STRIDE_INIT(4, 4, 4, 512, 20480),
  1, &_model_13_Resize_output_0_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _model_14_Concat_output_0_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 40, 40), AI_STRIDE_INIT(4, 4, 4, 768, 30720),
  1, &_model_14_Concat_output_0_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_Concat_output_0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 40, 40), AI_STRIDE_INIT(4, 4, 4, 384, 15360),
  1, &_model_15_Concat_output_0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_Split_output_0_num_or_size_splits, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_15_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_Split_output_0_output0, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_Split_output_0_output0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_Split_output_0_output1, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_Split_output_0_output1_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_act_Mul_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_15_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_model_15_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_weights, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 64), AI_STRIDE_INIT(4, 4, 768, 49152, 49152),
  1, &_model_15_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_act_Mul_output_0_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_15_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_15_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_model_15_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_weights, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 64), AI_STRIDE_INIT(4, 4, 384, 24576, 24576),
  1, &_model_15_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_15_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_15_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_15_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_15_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_15_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_15_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_15_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_act_Mul_output_0_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_16_act_Mul_output_0_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_act_Sigmoid_output_0_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_16_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_16_conv_Conv_output_0_bias_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_16_conv_Conv_output_0_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_scratch0, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_16_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_weights, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_16_conv_Conv_output_0_weights_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _model_17_Concat_output_0_output, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 20, 20), AI_STRIDE_INIT(4, 4, 4, 768, 15360),
  1, &_model_17_Concat_output_0_output_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_Concat_output_0_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 20, 20), AI_STRIDE_INIT(4, 4, 4, 768, 15360),
  1, &_model_18_Concat_output_0_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_Split_output_0_num_or_size_splits, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_18_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_Split_output_0_output0, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_Split_output_0_output0_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_Split_output_0_output1, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_Split_output_0_output1_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_act_Mul_output_0_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_bias, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_18_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_model_18_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_weights, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 128), AI_STRIDE_INIT(4, 4, 768, 98304, 98304),
  1, &_model_18_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_act_Mul_output_0_output, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_bias, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_18_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_output, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_18_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_model_18_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_weights, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 128), AI_STRIDE_INIT(4, 4, 768, 98304, 98304),
  1, &_model_18_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_18_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_18_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_18_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_18_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_18_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_18_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_18_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_act_Mul_output_0_output, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_19_act_Mul_output_0_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_act_Sigmoid_output_0_output, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_19_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_bias, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_19_conv_Conv_output_0_bias_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_19_conv_Conv_output_0_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_scratch0, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_19_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_weights, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 196608),
  1, &_model_19_conv_Conv_output_0_weights_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_act_Mul_output_0_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_1_act_Mul_output_0_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_act_Sigmoid_output_0_output, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_bias, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_scratch0, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 3, 3), AI_STRIDE_INIT(4, 4, 4, 64, 192),
  1, &_model_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_weights, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &_model_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  _model_20_Concat_output_0_output, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1536, 15360),
  1, &_model_20_Concat_output_0_output_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_Concat_output_0_output, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1536, 15360),
  1, &_model_21_Concat_output_0_output_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_Split_output_0_num_or_size_splits, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_21_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_Split_output_0_output0, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_Split_output_0_output0_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_Split_output_0_output1, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_Split_output_0_output1_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_act_Mul_output_0_output, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_bias, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_21_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_output, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &_model_21_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_weights, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 384, 1, 1, 256), AI_STRIDE_INIT(4, 4, 1536, 393216, 393216),
  1, &_model_21_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_act_Mul_output_0_output, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_bias, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_21_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_output, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_21_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &_model_21_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_weights, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 384, 1, 1, 256), AI_STRIDE_INIT(4, 4, 1536, 393216, 393216),
  1, &_model_21_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_21_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_21_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 196608),
  1, &_model_21_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_21_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_21_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_21_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 196608),
  1, &_model_21_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Add_1_output_0_output, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Add_1_output_0_output_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Add_2_output_0_output, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Add_2_output_0_output_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Concat_1_output_0_output, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_Concat_1_output_0_output_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Concat_2_output_0_output, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &_model_22_Concat_2_output_0_output_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Concat_output_0_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_Concat_output_0_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Constant_12_output_0, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Constant_12_output_0_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Constant_13_output_0, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Constant_13_output_0_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Constant_14_output_0_3D, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_22_Constant_14_output_0_3D_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Constant_15_output_0_3D, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_22_Constant_15_output_0_3D_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Div_1_output_0_output, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Div_1_output_0_output_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Mul_2_output_0_output, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &_model_22_Mul_2_output_0_output_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Sigmoid_output_0_output, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_Sigmoid_output_0_output_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Slice_1_output_0_output, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Slice_1_output_0_output_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Slice_output_0_output, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Slice_output_0_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Sub_1_output_0_output, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Sub_1_output_0_output_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_Sub_output_0_output, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_model_22_Sub_output_0_output_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Mul_output_0_output, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_0_act_Mul_output_0_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Mul_output_0_output, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_1_act_Mul_output_0_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_bias, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_bias_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_output, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_output_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_output0, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1600), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_output_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_weights, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_weights_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Mul_output_0_output, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_0_act_Mul_output_0_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 64), AI_STRIDE_INIT(4, 4, 512, 32768, 98304),
  1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Mul_output_0_output, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_1_act_Mul_output_0_output_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_bias, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_bias_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_output, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_output_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_output0, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 400), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_output_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_weights, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_weights_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Mul_output_0_output, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_0_act_Mul_output_0_output_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 3, 3), AI_STRIDE_INIT(4, 4, 4, 1024, 3072),
  1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 256, 3, 3, 64), AI_STRIDE_INIT(4, 4, 1024, 65536, 196608),
  1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Mul_output_0_output, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_1_act_Mul_output_0_output_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias, AI_STATIC,
  200, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_bias, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_bias_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_output, AI_STATIC,
  205, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_output_array, NULL)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_output0, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 100), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_output_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_weights, AI_STATIC,
  208, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_weights_array, NULL)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Mul_output_0_output, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_0_act_Mul_output_0_output_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Mul_output_0_output, AI_STATIC,
  215, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_1_act_Mul_output_0_output_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_bias, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_bias_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_output, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 40, 40), AI_STRIDE_INIT(4, 4, 4, 12, 480),
  1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_output_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_output0, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1600), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_output_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_weights, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 3), AI_STRIDE_INIT(4, 4, 256, 768, 768),
  1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_weights_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Mul_output_0_output, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_0_act_Mul_output_0_output_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 64), AI_STRIDE_INIT(4, 4, 512, 32768, 98304),
  1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Mul_output_0_output, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_1_act_Mul_output_0_output_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_bias, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_bias_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_output, AI_STATIC,
  239, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 20, 20), AI_STRIDE_INIT(4, 4, 4, 12, 240),
  1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_output_array, NULL)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_output0, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 400), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_output_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_weights, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 3), AI_STRIDE_INIT(4, 4, 256, 768, 768),
  1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_weights_array, NULL)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Mul_output_0_output, AI_STATIC,
  243, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_0_act_Mul_output_0_output_array, NULL)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output, AI_STATIC,
  244, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias_array, NULL)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output, AI_STATIC,
  246, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output_array, NULL)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0, AI_STATIC,
  247, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 3, 3), AI_STRIDE_INIT(4, 4, 4, 1024, 3072),
  1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 256, 3, 3, 64), AI_STRIDE_INIT(4, 4, 1024, 65536, 196608),
  1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights_array, NULL)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Mul_output_0_output, AI_STATIC,
  249, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_1_act_Mul_output_0_output_array, NULL)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output, AI_STATIC,
  252, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output_array, NULL)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0, AI_STATIC,
  253, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_bias, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_bias_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_output, AI_STATIC,
  256, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 10, 10), AI_STRIDE_INIT(4, 4, 4, 12, 120),
  1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_output_array, NULL)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_output0, AI_STATIC,
  257, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 100), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_output_array, NULL)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_weights, AI_STATIC,
  259, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 3), AI_STRIDE_INIT(4, 4, 256, 768, 768),
  1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_weights_array, NULL)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chfirst_output, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &_model_22_dfl_Reshape_1_output_0_to_chfirst_output_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chlast_output, AI_STATIC,
  261, 0x0,
  AI_SHAPE_INIT(4, 1, 2100, 4, 1), AI_STRIDE_INIT(4, 4, 4, 8400, 33600),
  1, &_model_22_dfl_Reshape_1_output_0_to_chlast_output_array, NULL)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chlast_output0, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 2100, 1, 4), AI_STRIDE_INIT(4, 4, 4, 8400, 8400),
  1, &_model_22_dfl_Reshape_1_output_0_to_chlast_output_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Reshape_output_0_to_chlast_output, AI_STATIC,
  263, 0x0,
  AI_SHAPE_INIT(4, 1, 2100, 1, 64), AI_STRIDE_INIT(4, 4, 4, 8400, 8400),
  1, &_model_22_dfl_Reshape_output_0_to_chlast_output_array, NULL)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Reshape_output_0_to_chlast_output0, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 2100, 16, 4), AI_STRIDE_INIT(4, 4, 4, 8400, 134400),
  1, &_model_22_dfl_Reshape_output_0_to_chlast_output_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Softmax_output_0_output, AI_STATIC,
  265, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 2100, 4), AI_STRIDE_INIT(4, 4, 4, 64, 134400),
  1, &_model_22_dfl_Softmax_output_0_output_array, NULL)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_Transpose_output_0_output, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 2100, 4), AI_STRIDE_INIT(4, 4, 4, 64, 134400),
  1, &_model_22_dfl_Transpose_output_0_output_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_bias, AI_STATIC,
  267, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_22_dfl_conv_Conv_output_0_bias_array, NULL)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_output, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 2100, 4), AI_STRIDE_INIT(4, 4, 4, 4, 8400),
  1, &_model_22_dfl_conv_Conv_output_0_output_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_scratch0, AI_STATIC,
  269, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_model_22_dfl_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_weights, AI_STATIC,
  270, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 1), AI_STRIDE_INIT(4, 4, 64, 64, 64),
  1, &_model_22_dfl_conv_Conv_output_0_weights_array, NULL)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_Concat_output_0_output, AI_STATIC,
  271, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 80, 80), AI_STRIDE_INIT(4, 4, 4, 192, 15360),
  1, &_model_2_Concat_output_0_output_array, NULL)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_Split_output_0_num_or_size_splits, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_2_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_Split_output_0_output0, AI_STATIC,
  273, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_Split_output_0_output0_array, NULL)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_Split_output_0_output1, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_Split_output_0_output1_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_act_Mul_output_0_output, AI_STATIC,
  275, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_bias, AI_STATIC,
  277, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_2_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_output, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  279, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_2_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_weights, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
  1, &_model_2_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_act_Mul_output_0_output, AI_STATIC,
  281, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_bias, AI_STATIC,
  283, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_2_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_output, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 80, 80), AI_STRIDE_INIT(4, 4, 4, 128, 10240),
  1, &_model_2_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  285, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_model_2_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_weights, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 1, 32), AI_STRIDE_INIT(4, 4, 192, 6144, 6144),
  1, &_model_2_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_Add_output_0_output, AI_STATIC,
  287, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_Add_output_0_output_array, NULL)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  288, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  289, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_model_2_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  291, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 3, 3), AI_STRIDE_INIT(4, 4, 4, 64, 192),
  1, &_model_2_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  293, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 4, 64, 1024, 3072),
  1, &_model_2_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  295, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_model_2_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  297, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &_model_2_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 3, 3), AI_STRIDE_INIT(4, 4, 4, 64, 192),
  1, &_model_2_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  299, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 4, 64, 1024, 3072),
  1, &_model_2_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_act_Mul_output_0_output, AI_STATIC,
  300, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_3_act_Mul_output_0_output_array, NULL)

/* Tensor #301 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_act_Sigmoid_output_0_output, AI_STATIC,
  301, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_3_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #302 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_bias, AI_STATIC,
  302, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_3_conv_Conv_output_0_bias_array, NULL)

/* Tensor #303 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_output, AI_STATIC,
  303, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_3_conv_Conv_output_0_output_array, NULL)

/* Tensor #304 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_scratch0, AI_STATIC,
  304, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_3_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #305 */
AI_TENSOR_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_weights, AI_STATIC,
  305, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 24576),
  1, &_model_3_conv_Conv_output_0_weights_array, NULL)

/* Tensor #306 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_Concat_output_0_output, AI_STATIC,
  306, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 40, 40), AI_STRIDE_INIT(4, 4, 4, 512, 20480),
  1, &_model_4_Concat_output_0_output_array, NULL)

/* Tensor #307 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_Split_output_0_num_or_size_splits, AI_STATIC,
  307, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_4_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #308 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_Split_output_0_output0, AI_STATIC,
  308, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_Split_output_0_output0_array, NULL)

/* Tensor #309 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_Split_output_0_output1, AI_STATIC,
  309, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_Split_output_0_output1_array, NULL)

/* Tensor #310 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_act_Mul_output_0_output, AI_STATIC,
  310, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #311 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  311, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #312 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_bias, AI_STATIC,
  312, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_4_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #313 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_output, AI_STATIC,
  313, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #314 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  314, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_4_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #315 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_weights, AI_STATIC,
  315, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_model_4_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #316 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_act_Mul_output_0_output, AI_STATIC,
  316, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #317 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  317, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #318 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_bias, AI_STATIC,
  318, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_4_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #319 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_output, AI_STATIC,
  319, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &_model_4_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #320 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  320, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_4_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #321 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_weights, AI_STATIC,
  321, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 64), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &_model_4_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #322 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_Add_output_0_output, AI_STATIC,
  322, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_Add_output_0_output_array, NULL)

/* Tensor #323 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  323, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #324 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  324, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #325 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  325, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_4_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #326 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  326, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #327 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  327, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_4_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #328 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  328, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_4_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #329 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  329, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #330 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  330, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #331 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  331, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_4_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #332 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  332, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #333 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  333, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_4_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #334 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  334, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_4_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #335 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_Add_output_0_output, AI_STATIC,
  335, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_Add_output_0_output_array, NULL)

/* Tensor #336 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Mul_output_0_output, AI_STATIC,
  336, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #337 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  337, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #338 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_bias, AI_STATIC,
  338, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_4_m_1_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #339 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_output, AI_STATIC,
  339, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #340 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  340, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_4_m_1_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #341 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_weights, AI_STATIC,
  341, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_4_m_1_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #342 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Mul_output_0_output, AI_STATIC,
  342, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #343 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  343, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #344 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_bias, AI_STATIC,
  344, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_4_m_1_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #345 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_output, AI_STATIC,
  345, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 4, 4, 128, 5120),
  1, &_model_4_m_1_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #346 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  346, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 3), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &_model_4_m_1_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #347 */
AI_TENSOR_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_weights, AI_STATIC,
  347, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &_model_4_m_1_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #348 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_act_Mul_output_0_output, AI_STATIC,
  348, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_5_act_Mul_output_0_output_array, NULL)

/* Tensor #349 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_act_Sigmoid_output_0_output, AI_STATIC,
  349, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_5_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #350 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_bias, AI_STATIC,
  350, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_5_conv_Conv_output_0_bias_array, NULL)

/* Tensor #351 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_output, AI_STATIC,
  351, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_5_conv_Conv_output_0_output_array, NULL)

/* Tensor #352 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_scratch0, AI_STATIC,
  352, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_5_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #353 */
AI_TENSOR_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_weights, AI_STATIC,
  353, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 4, 256, 32768, 98304),
  1, &_model_5_conv_Conv_output_0_weights_array, NULL)

/* Tensor #354 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_Concat_output_0_output, AI_STATIC,
  354, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 20, 20), AI_STRIDE_INIT(4, 4, 4, 1024, 20480),
  1, &_model_6_Concat_output_0_output_array, NULL)

/* Tensor #355 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_Split_output_0_num_or_size_splits, AI_STATIC,
  355, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_6_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #356 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_Split_output_0_output0, AI_STATIC,
  356, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_Split_output_0_output0_array, NULL)

/* Tensor #357 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_Split_output_0_output1, AI_STATIC,
  357, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_Split_output_0_output1_array, NULL)

/* Tensor #358 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_act_Mul_output_0_output, AI_STATIC,
  358, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #359 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  359, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #360 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_bias, AI_STATIC,
  360, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_6_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #361 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_output, AI_STATIC,
  361, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #362 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  362, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_6_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #363 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_weights, AI_STATIC,
  363, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &_model_6_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #364 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_act_Mul_output_0_output, AI_STATIC,
  364, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #365 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  365, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #366 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_bias, AI_STATIC,
  366, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_6_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #367 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_output, AI_STATIC,
  367, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 20), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &_model_6_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #368 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  368, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_6_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #369 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_weights, AI_STATIC,
  369, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 128), AI_STRIDE_INIT(4, 4, 1024, 131072, 131072),
  1, &_model_6_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #370 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_Add_output_0_output, AI_STATIC,
  370, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_Add_output_0_output_array, NULL)

/* Tensor #371 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  371, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #372 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  372, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #373 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  373, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_6_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #374 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  374, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #375 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  375, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_6_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #376 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  376, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_6_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #377 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  377, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #378 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  378, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #379 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  379, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_6_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #380 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  380, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #381 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  381, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_6_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #382 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  382, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_6_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #383 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_Add_output_0_output, AI_STATIC,
  383, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_Add_output_0_output_array, NULL)

/* Tensor #384 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Mul_output_0_output, AI_STATIC,
  384, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #385 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  385, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #386 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_bias, AI_STATIC,
  386, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_6_m_1_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #387 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_output, AI_STATIC,
  387, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #388 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  388, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_6_m_1_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #389 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_weights, AI_STATIC,
  389, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_6_m_1_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #390 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Mul_output_0_output, AI_STATIC,
  390, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #391 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  391, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #392 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_bias, AI_STATIC,
  392, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_6_m_1_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #393 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_output, AI_STATIC,
  393, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &_model_6_m_1_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #394 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  394, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &_model_6_m_1_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #395 */
AI_TENSOR_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_weights, AI_STATIC,
  395, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &_model_6_m_1_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #396 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_act_Mul_output_0_output, AI_STATIC,
  396, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_7_act_Mul_output_0_output_array, NULL)

/* Tensor #397 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_act_Sigmoid_output_0_output, AI_STATIC,
  397, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_7_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #398 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_bias, AI_STATIC,
  398, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_7_conv_Conv_output_0_bias_array, NULL)

/* Tensor #399 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_output, AI_STATIC,
  399, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_7_conv_Conv_output_0_output_array, NULL)

/* Tensor #400 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_scratch0, AI_STATIC,
  400, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_7_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #401 */
AI_TENSOR_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_weights, AI_STATIC,
  401, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 256), AI_STRIDE_INIT(4, 4, 512, 131072, 393216),
  1, &_model_7_conv_Conv_output_0_weights_array, NULL)

/* Tensor #402 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_Concat_output_0_output, AI_STATIC,
  402, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1536, 15360),
  1, &_model_8_Concat_output_0_output_array, NULL)

/* Tensor #403 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_Split_output_0_num_or_size_splits, AI_STATIC,
  403, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_8_Split_output_0_num_or_size_splits_array, NULL)

/* Tensor #404 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_Split_output_0_output0, AI_STATIC,
  404, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_Split_output_0_output0_array, NULL)

/* Tensor #405 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_Split_output_0_output1, AI_STATIC,
  405, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_Split_output_0_output1_array, NULL)

/* Tensor #406 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_act_Mul_output_0_output, AI_STATIC,
  406, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #407 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  407, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #408 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_bias, AI_STATIC,
  408, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_8_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #409 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_output, AI_STATIC,
  409, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #410 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  410, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_8_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #411 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_weights, AI_STATIC,
  411, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 4, 1024, 262144, 262144),
  1, &_model_8_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #412 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_act_Mul_output_0_output, AI_STATIC,
  412, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #413 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  413, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #414 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_bias, AI_STATIC,
  414, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_8_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #415 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_output, AI_STATIC,
  415, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_8_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #416 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  416, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &_model_8_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #417 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_weights, AI_STATIC,
  417, 0x0,
  AI_SHAPE_INIT(4, 384, 1, 1, 256), AI_STRIDE_INIT(4, 4, 1536, 393216, 393216),
  1, &_model_8_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #418 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_Add_output_0_output, AI_STATIC,
  418, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_Add_output_0_output_array, NULL)

/* Tensor #419 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Mul_output_0_output, AI_STATIC,
  419, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv1_act_Mul_output_0_output_array, NULL)

/* Tensor #420 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Sigmoid_output_0_output, AI_STATIC,
  420, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv1_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #421 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_bias, AI_STATIC,
  421, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_8_m_0_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #422 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_output, AI_STATIC,
  422, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #423 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  423, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_8_m_0_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #424 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_weights, AI_STATIC,
  424, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 196608),
  1, &_model_8_m_0_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #425 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Mul_output_0_output, AI_STATIC,
  425, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #426 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  426, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #427 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_bias, AI_STATIC,
  427, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_8_m_0_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #428 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_output, AI_STATIC,
  428, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_8_m_0_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #429 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  429, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &_model_8_m_0_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #430 */
AI_TENSOR_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_weights, AI_STATIC,
  430, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 196608),
  1, &_model_8_m_0_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #431 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_Concat_output_0_output, AI_STATIC,
  431, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 10, 10), AI_STRIDE_INIT(4, 4, 4, 2048, 20480),
  1, &_model_9_Concat_output_0_output_array, NULL)

/* Tensor #432 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_bias, AI_STATIC,
  432, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_9_cv1_conv_Conv_output_0_bias_array, NULL)

/* Tensor #433 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_output, AI_STATIC,
  433, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_9_cv1_conv_Conv_output_0_output_array, NULL)

/* Tensor #434 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_scratch0, AI_STATIC,
  434, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_9_cv1_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #435 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_weights, AI_STATIC,
  435, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 128), AI_STRIDE_INIT(4, 4, 1024, 131072, 131072),
  1, &_model_9_cv1_conv_Conv_output_0_weights_array, NULL)

/* Tensor #436 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_act_Mul_output_0_output, AI_STATIC,
  436, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_9_cv2_act_Mul_output_0_output_array, NULL)

/* Tensor #437 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_act_Sigmoid_output_0_output, AI_STATIC,
  437, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_9_cv2_act_Sigmoid_output_0_output_array, NULL)

/* Tensor #438 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_bias, AI_STATIC,
  438, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_model_9_cv2_conv_Conv_output_0_bias_array, NULL)

/* Tensor #439 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_output, AI_STATIC,
  439, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 10, 10), AI_STRIDE_INIT(4, 4, 4, 1024, 10240),
  1, &_model_9_cv2_conv_Conv_output_0_output_array, NULL)

/* Tensor #440 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_scratch0, AI_STATIC,
  440, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &_model_9_cv2_conv_Conv_output_0_scratch0_array, NULL)

/* Tensor #441 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_weights, AI_STATIC,
  441, 0x0,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 4, 2048, 524288, 524288),
  1, &_model_9_cv2_conv_Conv_output_0_weights_array, NULL)

/* Tensor #442 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_m_1_MaxPool_output_0_output, AI_STATIC,
  442, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_9_m_1_MaxPool_output_0_output_array, NULL)

/* Tensor #443 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_m_2_MaxPool_output_0_output, AI_STATIC,
  443, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_9_m_2_MaxPool_output_0_output_array, NULL)

/* Tensor #444 */
AI_TENSOR_OBJ_DECLARE(
  _model_9_m_MaxPool_output_0_output, AI_STATIC,
  444, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 4, 4, 512, 5120),
  1, &_model_9_m_MaxPool_output_0_output_array, NULL)

/* Tensor #445 */
AI_TENSOR_OBJ_DECLARE(
  images_Transpose_output, AI_STATIC,
  445, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 320, 320), AI_STRIDE_INIT(4, 4, 4, 12, 3840),
  1, &images_Transpose_output_array, NULL)

/* Tensor #446 */
AI_TENSOR_OBJ_DECLARE(
  images_output, AI_STATIC,
  446, 0x0,
  AI_SHAPE_INIT(4, 1, 320, 320, 3), AI_STRIDE_INIT(4, 4, 4, 1280, 409600),
  1, &images_output_array, NULL)

/* Tensor #447 */
AI_TENSOR_OBJ_DECLARE(
  output0_Transpose_0_output, AI_STATIC,
  447, 0x0,
  AI_SHAPE_INIT(4, 1, 2100, 1, 7), AI_STRIDE_INIT(4, 4, 4, 8400, 8400),
  1, &output0_Transpose_0_output_array, NULL)

/* Tensor #448 */
AI_TENSOR_OBJ_DECLARE(
  output0_output, AI_STATIC,
  448, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 2100), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &output0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  output0_Transpose_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output0_Transpose_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output0_Transpose_0_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &output0_Transpose_0_chain,
  NULL, &output0_Transpose_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Mul_2_output_0_output, &_model_22_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output0_layer, 231,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &output0_chain,
  NULL, &output0_Transpose_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Concat_2_output_0_output, &_model_22_Constant_15_output_0_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Mul_2_output_0_layer, 230,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Mul_2_output_0_chain,
  NULL, &output0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Concat_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Div_1_output_0_output, &_model_22_Sub_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Concat_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Concat_2_output_0_layer, 229,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_22_Concat_2_output_0_chain,
  NULL, &_model_22_Mul_2_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Div_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Add_2_output_0_output, &_model_22_Constant_14_output_0_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Div_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Div_1_output_0_layer, 228,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Div_1_output_0_chain,
  NULL, &_model_22_Concat_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Add_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Sub_output_0_output, &_model_22_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Add_2_output_0_layer, 226,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Add_2_output_0_chain,
  NULL, &_model_22_Div_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Sub_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Add_1_output_0_output, &_model_22_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Sub_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Sub_1_output_0_layer, 227,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Sub_1_output_0_chain,
  NULL, &_model_22_Add_2_output_0_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Constant_12_output_0, &_model_22_Slice_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Sub_output_0_layer, 224,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Sub_output_0_chain,
  NULL, &_model_22_Sub_1_output_0_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_Constant_13_output_0, &_model_22_Slice_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Add_1_output_0_layer, 225,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_Add_1_output_0_chain,
  NULL, &_model_22_Sub_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 _model_22_Slice_output_0_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_output_0_axes, AI_ARRAY_FORMAT_U8,
    _model_22_Slice_output_0_axes_data, _model_22_Slice_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _model_22_Slice_output_0_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_output_0_starts, AI_ARRAY_FORMAT_S16,
    _model_22_Slice_output_0_starts_data, _model_22_Slice_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _model_22_Slice_output_0_ends_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_output_0_ends, AI_ARRAY_FORMAT_S16,
    _model_22_Slice_output_0_ends_data, _model_22_Slice_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Slice_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_1_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Slice_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Slice_output_0_layer, 222,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_model_22_Slice_output_0_chain,
  NULL, &_model_22_Add_1_output_0_layer, AI_STATIC, 
  .axes = &_model_22_Slice_output_0_axes, 
  .starts = &_model_22_Slice_output_0_starts, 
  .ends = &_model_22_Slice_output_0_ends, 
)


AI_STATIC_CONST ai_u8 _model_22_Slice_1_output_0_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_1_output_0_axes, AI_ARRAY_FORMAT_U8,
    _model_22_Slice_1_output_0_axes_data, _model_22_Slice_1_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _model_22_Slice_1_output_0_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_1_output_0_starts, AI_ARRAY_FORMAT_S16,
    _model_22_Slice_1_output_0_starts_data, _model_22_Slice_1_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _model_22_Slice_1_output_0_ends_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    _model_22_Slice_1_output_0_ends, AI_ARRAY_FORMAT_S16,
    _model_22_Slice_1_output_0_ends_data, _model_22_Slice_1_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Slice_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_1_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Slice_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Slice_1_output_0_layer, 223,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_model_22_Slice_1_output_0_chain,
  NULL, &_model_22_Slice_output_0_layer, AI_STATIC, 
  .axes = &_model_22_Slice_1_output_0_axes, 
  .starts = &_model_22_Slice_1_output_0_starts, 
  .ends = &_model_22_Slice_1_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chfirst_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_1_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_1_output_0_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chfirst_layer, 221,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_model_22_dfl_Reshape_1_output_0_to_chfirst_chain,
  NULL, &_model_22_Slice_1_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_1_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_Reshape_1_output_0_to_chlast_layer, 221,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_model_22_dfl_Reshape_1_output_0_to_chlast_chain,
  NULL, &_model_22_dfl_Reshape_1_output_0_to_chfirst_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Softmax_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_dfl_conv_Conv_output_0_weights, &_model_22_dfl_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_dfl_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_conv_Conv_output_0_layer, 220,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_dfl_conv_Conv_output_0_chain,
  NULL, &_model_22_dfl_Reshape_1_output_0_to_chlast_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_Softmax_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Softmax_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_Softmax_output_0_layer, 219,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &_model_22_dfl_Softmax_output_0_chain,
  NULL, &_model_22_dfl_conv_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_Transpose_output_0_layer, 218,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_model_22_dfl_Transpose_output_0_chain,
  NULL, &_model_22_dfl_Softmax_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_dfl_Reshape_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_dfl_Reshape_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_dfl_Reshape_output_0_to_chlast_layer, 216,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_model_22_dfl_Reshape_output_0_to_chlast_chain,
  NULL, &_model_22_dfl_Transpose_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_0_cv2_0_2_Conv_output_0_output0, &_model_22_cv2_1_cv2_1_2_Conv_output_0_output0, &_model_22_cv2_2_cv2_2_2_Conv_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Concat_output_0_layer, 214,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_22_Concat_output_0_chain,
  NULL, &_model_22_dfl_Reshape_output_0_to_chlast_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_2_cv2_2_2_Conv_output_0_weights, &_model_22_cv2_2_cv2_2_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_2_Conv_output_0_layer, 210,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_2_cv2_2_2_Conv_output_0_chain,
  NULL, &_model_22_Concat_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output, &_model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Mul_output_0_layer, 208,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_2_cv2_2_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_layer, 206,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_layer, 204,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output, &_model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Mul_output_0_layer, 202,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_2_cv2_2_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_layer, 200,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_layer, 198,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Sigmoid_output_0_layer, 217,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_2_cv2_2_0_conv_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_Concat_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_0_cv3_0_2_Conv_output_0_output0, &_model_22_cv3_1_cv3_1_2_Conv_output_0_output0, &_model_22_cv3_2_cv3_2_2_Conv_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_Concat_1_output_0_layer, 215,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_22_Concat_1_output_0_chain,
  NULL, &_model_22_Sigmoid_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_2_cv3_2_2_Conv_output_0_weights, &_model_22_cv3_2_cv3_2_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_2_Conv_output_0_layer, 211,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_2_cv3_2_2_Conv_output_0_chain,
  NULL, &_model_22_Concat_1_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output, &_model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Mul_output_0_layer, 209,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_2_cv3_2_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_layer, 207,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_layer, 205,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output, &_model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Mul_output_0_layer, 203,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_2_cv3_2_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_layer, 201,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_layer, 199,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_cv2_conv_Conv_output_0_output, &_model_21_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv2_act_Mul_output_0_layer, 197,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_21_cv2_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_2_cv3_2_0_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv2_act_Sigmoid_output_0_layer, 196,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_21_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_21_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_21_cv2_conv_Conv_output_0_weights, &_model_21_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv2_conv_Conv_output_0_layer, 195,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_21_cv2_conv_Conv_output_0_chain,
  NULL, &_model_21_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_21_Split_output_0_output0, &_model_21_Split_output_0_output1, &_model_21_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_Concat_output_0_layer, 194,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_21_Concat_output_0_chain,
  NULL, &_model_21_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_m_0_cv2_conv_Conv_output_0_output, &_model_21_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Mul_output_0_layer, 193,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_21_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_21_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv2_act_Sigmoid_output_0_layer, 192,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_21_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_21_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_21_m_0_cv2_conv_Conv_output_0_weights, &_model_21_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv2_conv_Conv_output_0_layer, 191,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_21_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_21_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_m_0_cv1_conv_Conv_output_0_output, &_model_21_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Mul_output_0_layer, 190,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_21_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_21_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv1_act_Sigmoid_output_0_layer, 189,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_21_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_21_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_21_m_0_cv1_conv_Conv_output_0_weights, &_model_21_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_21_m_0_cv1_conv_Conv_output_0_layer, 188,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_21_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_21_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_Split_output_0_output0, &_model_21_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_Split_output_0_layer, 185,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_21_Split_output_0_chain,
  NULL, &_model_21_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 100, 
  .outer_elems_stride = 1024, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_cv1_conv_Conv_output_0_output, &_model_21_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv1_act_Mul_output_0_layer, 182,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_21_cv1_act_Mul_output_0_chain,
  NULL, &_model_21_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv1_act_Sigmoid_output_0_layer, 179,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_21_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_21_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_20_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_21_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_21_cv1_conv_Conv_output_0_weights, &_model_21_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_21_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_21_cv1_conv_Conv_output_0_layer, 176,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_21_cv1_conv_Conv_output_0_chain,
  NULL, &_model_21_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_20_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_19_act_Mul_output_0_output, &_model_9_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_20_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_20_Concat_output_0_layer, 173,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_20_Concat_output_0_chain,
  NULL, &_model_21_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_19_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_19_conv_Conv_output_0_output, &_model_19_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_19_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_19_act_Mul_output_0_layer, 170,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_19_act_Mul_output_0_chain,
  NULL, &_model_20_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_19_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_19_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_19_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_19_act_Sigmoid_output_0_layer, 167,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_19_act_Sigmoid_output_0_chain,
  NULL, &_model_19_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_19_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_19_conv_Conv_output_0_weights, &_model_19_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_19_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_19_conv_Conv_output_0_layer, 164,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_19_conv_Conv_output_0_chain,
  NULL, &_model_19_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_1_cv2_1_2_Conv_output_0_weights, &_model_22_cv2_1_cv2_1_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_2_Conv_output_0_layer, 183,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_1_cv2_1_2_Conv_output_0_chain,
  NULL, &_model_19_conv_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output, &_model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Mul_output_0_layer, 180,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_1_cv2_1_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_layer, 177,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_layer, 174,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output, &_model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Mul_output_0_layer, 171,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_1_cv2_1_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_layer, 168,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_layer, 165,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_1_cv3_1_2_Conv_output_0_weights, &_model_22_cv3_1_cv3_1_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_2_Conv_output_0_layer, 184,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_1_cv3_1_2_Conv_output_0_chain,
  NULL, &_model_22_cv2_1_cv2_1_0_conv_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output, &_model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Mul_output_0_layer, 181,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_1_cv3_1_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_layer, 178,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_layer, 175,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output, &_model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Mul_output_0_layer, 172,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_1_cv3_1_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_layer, 169,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_layer, 166,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_cv2_conv_Conv_output_0_output, &_model_18_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv2_act_Mul_output_0_layer, 163,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_18_cv2_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_1_cv3_1_0_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv2_act_Sigmoid_output_0_layer, 162,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_18_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_18_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_18_cv2_conv_Conv_output_0_weights, &_model_18_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv2_conv_Conv_output_0_layer, 161,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_18_cv2_conv_Conv_output_0_chain,
  NULL, &_model_18_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_18_Split_output_0_output0, &_model_18_Split_output_0_output1, &_model_18_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_Concat_output_0_layer, 160,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_18_Concat_output_0_chain,
  NULL, &_model_18_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_m_0_cv2_conv_Conv_output_0_output, &_model_18_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Mul_output_0_layer, 159,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_18_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_18_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv2_act_Sigmoid_output_0_layer, 158,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_18_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_18_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_18_m_0_cv2_conv_Conv_output_0_weights, &_model_18_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv2_conv_Conv_output_0_layer, 157,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_18_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_18_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_m_0_cv1_conv_Conv_output_0_output, &_model_18_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Mul_output_0_layer, 156,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_18_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_18_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv1_act_Sigmoid_output_0_layer, 155,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_18_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_18_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_18_m_0_cv1_conv_Conv_output_0_weights, &_model_18_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_18_m_0_cv1_conv_Conv_output_0_layer, 154,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_18_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_18_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_Split_output_0_output0, &_model_18_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_Split_output_0_layer, 151,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_18_Split_output_0_chain,
  NULL, &_model_18_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 400, 
  .outer_elems_stride = 512, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_cv1_conv_Conv_output_0_output, &_model_18_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv1_act_Mul_output_0_layer, 148,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_18_cv1_act_Mul_output_0_chain,
  NULL, &_model_18_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv1_act_Sigmoid_output_0_layer, 145,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_18_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_18_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_17_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_18_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_18_cv1_conv_Conv_output_0_weights, &_model_18_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_18_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_18_cv1_conv_Conv_output_0_layer, 142,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_18_cv1_conv_Conv_output_0_chain,
  NULL, &_model_18_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_17_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_16_act_Mul_output_0_output, &_model_12_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_17_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_17_Concat_output_0_layer, 139,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_17_Concat_output_0_chain,
  NULL, &_model_18_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_16_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_16_conv_Conv_output_0_output, &_model_16_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_16_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_16_act_Mul_output_0_layer, 136,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_16_act_Mul_output_0_chain,
  NULL, &_model_17_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_16_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_16_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_16_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_16_act_Sigmoid_output_0_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_16_act_Sigmoid_output_0_chain,
  NULL, &_model_16_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_16_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_16_conv_Conv_output_0_weights, &_model_16_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_16_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_16_conv_Conv_output_0_layer, 130,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_16_conv_Conv_output_0_chain,
  NULL, &_model_16_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_0_cv2_0_2_Conv_output_0_weights, &_model_22_cv2_0_cv2_0_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_2_Conv_output_0_layer, 149,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_0_cv2_0_2_Conv_output_0_chain,
  NULL, &_model_16_conv_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output, &_model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Mul_output_0_layer, 146,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_0_cv2_0_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_layer, 143,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_layer, 140,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output, &_model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Mul_output_0_layer, 137,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv2_0_cv2_0_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_layer, 131,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_0_cv3_0_2_Conv_output_0_weights, &_model_22_cv3_0_cv3_0_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_2_Conv_output_0_layer, 150,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_0_cv3_0_2_Conv_output_0_chain,
  NULL, &_model_22_cv2_0_cv2_0_0_conv_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output, &_model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Mul_output_0_layer, 147,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_0_cv3_0_1_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_layer, 144,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_layer, 141,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output, &_model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Mul_output_0_layer, 138,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_22_cv3_0_cv3_0_0_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_layer, 135,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_layer, 132,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_cv2_conv_Conv_output_0_output, &_model_15_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv2_act_Mul_output_0_layer, 129,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_15_cv2_act_Mul_output_0_chain,
  NULL, &_model_22_cv3_0_cv3_0_0_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv2_act_Sigmoid_output_0_layer, 128,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_15_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_15_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_15_cv2_conv_Conv_output_0_weights, &_model_15_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv2_conv_Conv_output_0_layer, 127,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_15_cv2_conv_Conv_output_0_chain,
  NULL, &_model_15_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_15_Split_output_0_output0, &_model_15_Split_output_0_output1, &_model_15_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_Concat_output_0_layer, 126,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_15_Concat_output_0_chain,
  NULL, &_model_15_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_m_0_cv2_conv_Conv_output_0_output, &_model_15_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Mul_output_0_layer, 125,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_15_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_15_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv2_act_Sigmoid_output_0_layer, 124,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_15_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_15_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_15_m_0_cv2_conv_Conv_output_0_weights, &_model_15_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv2_conv_Conv_output_0_layer, 123,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_15_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_15_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_m_0_cv1_conv_Conv_output_0_output, &_model_15_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Mul_output_0_layer, 122,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_15_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_15_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv1_act_Sigmoid_output_0_layer, 121,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_15_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_15_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_15_m_0_cv1_conv_Conv_output_0_weights, &_model_15_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_15_m_0_cv1_conv_Conv_output_0_layer, 120,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_15_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_15_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_Split_output_0_output0, &_model_15_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_Split_output_0_layer, 119,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_15_Split_output_0_chain,
  NULL, &_model_15_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 1600, 
  .outer_elems_stride = 256, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_cv1_conv_Conv_output_0_output, &_model_15_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv1_act_Mul_output_0_layer, 118,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_15_cv1_act_Mul_output_0_chain,
  NULL, &_model_15_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv1_act_Sigmoid_output_0_layer, 117,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_15_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_15_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_14_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_15_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_15_cv1_conv_Conv_output_0_weights, &_model_15_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_15_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_15_cv1_conv_Conv_output_0_layer, 116,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_15_cv1_conv_Conv_output_0_chain,
  NULL, &_model_15_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_14_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_13_Resize_output_0_output, &_model_4_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_14_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_14_Concat_output_0_layer, 115,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_14_Concat_output_0_chain,
  NULL, &_model_15_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float _model_13_Resize_output_0_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_13_Resize_output_0_scales, AI_ARRAY_FORMAT_FLOAT,
    _model_13_Resize_output_0_scales_data, _model_13_Resize_output_0_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_13_Resize_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_13_Resize_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_13_Resize_output_0_layer, 114,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &_model_13_Resize_output_0_chain,
  NULL, &_model_14_Concat_output_0_layer, AI_STATIC, 
  .scales = &_model_13_Resize_output_0_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_cv2_conv_Conv_output_0_output, &_model_12_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv2_act_Mul_output_0_layer, 113,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_12_cv2_act_Mul_output_0_chain,
  NULL, &_model_13_Resize_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv2_act_Sigmoid_output_0_layer, 112,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_12_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_12_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_12_cv2_conv_Conv_output_0_weights, &_model_12_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv2_conv_Conv_output_0_layer, 111,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_12_cv2_conv_Conv_output_0_chain,
  NULL, &_model_12_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_12_Split_output_0_output0, &_model_12_Split_output_0_output1, &_model_12_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_Concat_output_0_layer, 110,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_12_Concat_output_0_chain,
  NULL, &_model_12_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_m_0_cv2_conv_Conv_output_0_output, &_model_12_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Mul_output_0_layer, 109,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_12_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_12_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv2_act_Sigmoid_output_0_layer, 108,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_12_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_12_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_12_m_0_cv2_conv_Conv_output_0_weights, &_model_12_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv2_conv_Conv_output_0_layer, 107,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_12_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_12_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_m_0_cv1_conv_Conv_output_0_output, &_model_12_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Mul_output_0_layer, 106,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_12_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_12_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv1_act_Sigmoid_output_0_layer, 105,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_12_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_12_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_12_m_0_cv1_conv_Conv_output_0_weights, &_model_12_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_12_m_0_cv1_conv_Conv_output_0_layer, 104,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_12_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_12_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_Split_output_0_output0, &_model_12_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_Split_output_0_layer, 103,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_12_Split_output_0_chain,
  NULL, &_model_12_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 400, 
  .outer_elems_stride = 512, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_cv1_conv_Conv_output_0_output, &_model_12_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv1_act_Mul_output_0_layer, 102,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_12_cv1_act_Mul_output_0_chain,
  NULL, &_model_12_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv1_act_Sigmoid_output_0_layer, 101,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_12_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_12_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_11_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_12_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_12_cv1_conv_Conv_output_0_weights, &_model_12_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_12_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_12_cv1_conv_Conv_output_0_layer, 100,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_12_cv1_conv_Conv_output_0_chain,
  NULL, &_model_12_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_11_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_10_Resize_output_0_output, &_model_6_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_11_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_11_Concat_output_0_layer, 99,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_11_Concat_output_0_chain,
  NULL, &_model_12_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float _model_10_Resize_output_0_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_10_Resize_output_0_scales, AI_ARRAY_FORMAT_FLOAT,
    _model_10_Resize_output_0_scales_data, _model_10_Resize_output_0_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_10_Resize_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_10_Resize_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_10_Resize_output_0_layer, 98,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &_model_10_Resize_output_0_chain,
  NULL, &_model_11_Concat_output_0_layer, AI_STATIC, 
  .scales = &_model_10_Resize_output_0_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_9_cv2_conv_Conv_output_0_output, &_model_9_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_cv2_act_Mul_output_0_layer, 97,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_9_cv2_act_Mul_output_0_chain,
  NULL, &_model_10_Resize_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_cv2_act_Sigmoid_output_0_layer, 96,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_9_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_9_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_9_cv2_conv_Conv_output_0_weights, &_model_9_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_9_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_9_cv2_conv_Conv_output_0_layer, 95,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_9_cv2_conv_Conv_output_0_chain,
  NULL, &_model_9_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 4, &_model_9_cv1_conv_Conv_output_0_output, &_model_9_m_MaxPool_output_0_output, &_model_9_m_1_MaxPool_output_0_output, &_model_9_m_2_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_Concat_output_0_layer, 94,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_9_Concat_output_0_chain,
  NULL, &_model_9_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_m_2_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_m_1_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_m_2_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_m_2_MaxPool_output_0_layer, 93,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_9_m_2_MaxPool_output_0_chain,
  NULL, &_model_9_Concat_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 5), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 2, 2, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_m_1_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_m_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_m_1_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_m_1_MaxPool_output_0_layer, 92,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_9_m_1_MaxPool_output_0_chain,
  NULL, &_model_9_m_2_MaxPool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 5), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 2, 2, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_m_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_m_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_9_m_MaxPool_output_0_layer, 91,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_9_m_MaxPool_output_0_chain,
  NULL, &_model_9_m_1_MaxPool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 5), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 2, 2, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_9_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_9_cv1_conv_Conv_output_0_weights, &_model_9_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_9_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_9_cv1_conv_Conv_output_0_layer, 90,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_9_cv1_conv_Conv_output_0_chain,
  NULL, &_model_9_m_MaxPool_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_cv2_conv_Conv_output_0_output, &_model_8_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv2_act_Mul_output_0_layer, 89,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_8_cv2_act_Mul_output_0_chain,
  NULL, &_model_9_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv2_act_Sigmoid_output_0_layer, 88,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_8_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_8_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_8_cv2_conv_Conv_output_0_weights, &_model_8_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv2_conv_Conv_output_0_layer, 87,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_8_cv2_conv_Conv_output_0_chain,
  NULL, &_model_8_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_8_Split_output_0_output0, &_model_8_Split_output_0_output1, &_model_8_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_Concat_output_0_layer, 86,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_8_Concat_output_0_chain,
  NULL, &_model_8_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_Split_output_0_output1, &_model_8_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_Add_output_0_layer, 85,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_8_m_0_Add_output_0_chain,
  NULL, &_model_8_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_m_0_cv2_conv_Conv_output_0_output, &_model_8_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Mul_output_0_layer, 84,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_8_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_8_m_0_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv2_act_Sigmoid_output_0_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_8_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_8_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_8_m_0_cv2_conv_Conv_output_0_weights, &_model_8_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv2_conv_Conv_output_0_layer, 82,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_8_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_8_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_m_0_cv1_conv_Conv_output_0_output, &_model_8_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Mul_output_0_layer, 81,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_8_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_8_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv1_act_Sigmoid_output_0_layer, 80,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_8_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_8_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_8_m_0_cv1_conv_Conv_output_0_weights, &_model_8_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_8_m_0_cv1_conv_Conv_output_0_layer, 79,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_8_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_8_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_Split_output_0_output0, &_model_8_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_Split_output_0_layer, 78,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_8_Split_output_0_chain,
  NULL, &_model_8_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 100, 
  .outer_elems_stride = 1024, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_cv1_conv_Conv_output_0_output, &_model_8_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv1_act_Mul_output_0_layer, 77,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_8_cv1_act_Mul_output_0_chain,
  NULL, &_model_8_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv1_act_Sigmoid_output_0_layer, 76,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_8_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_8_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_7_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_8_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_8_cv1_conv_Conv_output_0_weights, &_model_8_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_8_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_8_cv1_conv_Conv_output_0_layer, 75,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_8_cv1_conv_Conv_output_0_chain,
  NULL, &_model_8_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_7_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_7_conv_Conv_output_0_output, &_model_7_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_7_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_7_act_Mul_output_0_layer, 74,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_7_act_Mul_output_0_chain,
  NULL, &_model_8_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_7_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_7_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_7_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_7_act_Sigmoid_output_0_layer, 73,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_7_act_Sigmoid_output_0_chain,
  NULL, &_model_7_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_7_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_7_conv_Conv_output_0_weights, &_model_7_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_7_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_7_conv_Conv_output_0_layer, 72,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_7_conv_Conv_output_0_chain,
  NULL, &_model_7_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_cv2_conv_Conv_output_0_output, &_model_6_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv2_act_Mul_output_0_layer, 71,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_cv2_act_Mul_output_0_chain,
  NULL, &_model_7_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv2_act_Sigmoid_output_0_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_6_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_cv2_conv_Conv_output_0_weights, &_model_6_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv2_conv_Conv_output_0_layer, 69,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_cv2_conv_Conv_output_0_chain,
  NULL, &_model_6_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 4, &_model_6_Split_output_0_output0, &_model_6_Split_output_0_output1, &_model_6_m_0_Add_output_0_output, &_model_6_m_1_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_Concat_output_0_layer, 68,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_6_Concat_output_0_chain,
  NULL, &_model_6_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_0_Add_output_0_output, &_model_6_m_1_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_Add_output_0_layer, 67,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_1_Add_output_0_chain,
  NULL, &_model_6_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_1_cv2_conv_Conv_output_0_output, &_model_6_m_1_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Mul_output_0_layer, 66,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_1_cv2_act_Mul_output_0_chain,
  NULL, &_model_6_m_1_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv2_act_Sigmoid_output_0_layer, 65,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_m_1_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_6_m_1_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_m_1_cv2_conv_Conv_output_0_weights, &_model_6_m_1_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_1_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv2_conv_Conv_output_0_layer, 64,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_m_1_cv2_conv_Conv_output_0_chain,
  NULL, &_model_6_m_1_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_1_cv1_conv_Conv_output_0_output, &_model_6_m_1_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Mul_output_0_layer, 63,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_1_cv1_act_Mul_output_0_chain,
  NULL, &_model_6_m_1_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv1_act_Sigmoid_output_0_layer, 62,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_m_1_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_6_m_1_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_1_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_m_1_cv1_conv_Conv_output_0_weights, &_model_6_m_1_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_1_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_1_cv1_conv_Conv_output_0_layer, 61,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_m_1_cv1_conv_Conv_output_0_chain,
  NULL, &_model_6_m_1_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_Split_output_0_output1, &_model_6_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_Add_output_0_layer, 60,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_0_Add_output_0_chain,
  NULL, &_model_6_m_1_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_0_cv2_conv_Conv_output_0_output, &_model_6_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Mul_output_0_layer, 59,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_6_m_0_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv2_act_Sigmoid_output_0_layer, 58,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_6_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_m_0_cv2_conv_Conv_output_0_weights, &_model_6_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv2_conv_Conv_output_0_layer, 57,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_6_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_0_cv1_conv_Conv_output_0_output, &_model_6_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Mul_output_0_layer, 56,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_6_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv1_act_Sigmoid_output_0_layer, 55,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_6_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_m_0_cv1_conv_Conv_output_0_weights, &_model_6_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_m_0_cv1_conv_Conv_output_0_layer, 54,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_6_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_Split_output_0_output0, &_model_6_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_Split_output_0_layer, 53,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_6_Split_output_0_chain,
  NULL, &_model_6_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 400, 
  .outer_elems_stride = 512, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_cv1_conv_Conv_output_0_output, &_model_6_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv1_act_Mul_output_0_layer, 52,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_6_cv1_act_Mul_output_0_chain,
  NULL, &_model_6_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv1_act_Sigmoid_output_0_layer, 51,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_6_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_6_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_5_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_6_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_6_cv1_conv_Conv_output_0_weights, &_model_6_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_6_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_6_cv1_conv_Conv_output_0_layer, 50,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_6_cv1_conv_Conv_output_0_chain,
  NULL, &_model_6_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_5_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_5_conv_Conv_output_0_output, &_model_5_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_5_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_5_act_Mul_output_0_layer, 49,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_5_act_Mul_output_0_chain,
  NULL, &_model_6_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_5_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_5_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_5_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_5_act_Sigmoid_output_0_layer, 48,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_5_act_Sigmoid_output_0_chain,
  NULL, &_model_5_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_5_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_5_conv_Conv_output_0_weights, &_model_5_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_5_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_5_conv_Conv_output_0_layer, 47,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_5_conv_Conv_output_0_chain,
  NULL, &_model_5_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_cv2_conv_Conv_output_0_output, &_model_4_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv2_act_Mul_output_0_layer, 46,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_cv2_act_Mul_output_0_chain,
  NULL, &_model_5_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv2_act_Sigmoid_output_0_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_4_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_cv2_conv_Conv_output_0_weights, &_model_4_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv2_conv_Conv_output_0_layer, 44,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_cv2_conv_Conv_output_0_chain,
  NULL, &_model_4_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 4, &_model_4_Split_output_0_output0, &_model_4_Split_output_0_output1, &_model_4_m_0_Add_output_0_output, &_model_4_m_1_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_Concat_output_0_layer, 43,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_4_Concat_output_0_chain,
  NULL, &_model_4_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_0_Add_output_0_output, &_model_4_m_1_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_Add_output_0_layer, 42,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_1_Add_output_0_chain,
  NULL, &_model_4_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_1_cv2_conv_Conv_output_0_output, &_model_4_m_1_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Mul_output_0_layer, 41,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_1_cv2_act_Mul_output_0_chain,
  NULL, &_model_4_m_1_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv2_act_Sigmoid_output_0_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_m_1_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_4_m_1_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_m_1_cv2_conv_Conv_output_0_weights, &_model_4_m_1_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_1_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv2_conv_Conv_output_0_layer, 39,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_m_1_cv2_conv_Conv_output_0_chain,
  NULL, &_model_4_m_1_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_1_cv1_conv_Conv_output_0_output, &_model_4_m_1_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Mul_output_0_layer, 38,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_1_cv1_act_Mul_output_0_chain,
  NULL, &_model_4_m_1_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv1_act_Sigmoid_output_0_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_m_1_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_4_m_1_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_1_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_m_1_cv1_conv_Conv_output_0_weights, &_model_4_m_1_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_1_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_1_cv1_conv_Conv_output_0_layer, 36,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_m_1_cv1_conv_Conv_output_0_chain,
  NULL, &_model_4_m_1_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_Split_output_0_output1, &_model_4_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_Add_output_0_layer, 35,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_0_Add_output_0_chain,
  NULL, &_model_4_m_1_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_0_cv2_conv_Conv_output_0_output, &_model_4_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Mul_output_0_layer, 34,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_4_m_0_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv2_act_Sigmoid_output_0_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_4_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_m_0_cv2_conv_Conv_output_0_weights, &_model_4_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv2_conv_Conv_output_0_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_4_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_0_cv1_conv_Conv_output_0_output, &_model_4_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Mul_output_0_layer, 31,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_4_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv1_act_Sigmoid_output_0_layer, 30,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_4_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_m_0_cv1_conv_Conv_output_0_weights, &_model_4_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_m_0_cv1_conv_Conv_output_0_layer, 29,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_4_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_Split_output_0_output0, &_model_4_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_Split_output_0_layer, 28,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_4_Split_output_0_chain,
  NULL, &_model_4_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 1600, 
  .outer_elems_stride = 256, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_cv1_conv_Conv_output_0_output, &_model_4_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv1_act_Mul_output_0_layer, 27,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_4_cv1_act_Mul_output_0_chain,
  NULL, &_model_4_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv1_act_Sigmoid_output_0_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_4_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_4_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_3_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_4_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_4_cv1_conv_Conv_output_0_weights, &_model_4_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_4_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_4_cv1_conv_Conv_output_0_layer, 25,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_4_cv1_conv_Conv_output_0_chain,
  NULL, &_model_4_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_3_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_3_conv_Conv_output_0_output, &_model_3_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_3_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_3_act_Mul_output_0_layer, 24,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_3_act_Mul_output_0_chain,
  NULL, &_model_4_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_3_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_3_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_3_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_3_act_Sigmoid_output_0_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_3_act_Sigmoid_output_0_chain,
  NULL, &_model_3_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_3_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_3_conv_Conv_output_0_weights, &_model_3_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_3_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_3_conv_Conv_output_0_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_3_conv_Conv_output_0_chain,
  NULL, &_model_3_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_cv2_conv_Conv_output_0_output, &_model_2_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv2_act_Mul_output_0_layer, 21,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_2_cv2_act_Mul_output_0_chain,
  NULL, &_model_3_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv2_act_Sigmoid_output_0_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_2_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_2_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_2_cv2_conv_Conv_output_0_weights, &_model_2_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv2_conv_Conv_output_0_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_2_cv2_conv_Conv_output_0_chain,
  NULL, &_model_2_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_2_Split_output_0_output0, &_model_2_Split_output_0_output1, &_model_2_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_Concat_output_0_layer, 18,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_2_Concat_output_0_chain,
  NULL, &_model_2_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_Split_output_0_output1, &_model_2_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_Add_output_0_layer, 17,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_2_m_0_Add_output_0_chain,
  NULL, &_model_2_Concat_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_m_0_cv2_conv_Conv_output_0_output, &_model_2_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv2_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Mul_output_0_layer, 16,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_2_m_0_cv2_act_Mul_output_0_chain,
  NULL, &_model_2_m_0_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv2_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv2_act_Sigmoid_output_0_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_2_m_0_cv2_act_Sigmoid_output_0_chain,
  NULL, &_model_2_m_0_cv2_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv2_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_2_m_0_cv2_conv_Conv_output_0_weights, &_model_2_m_0_cv2_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_m_0_cv2_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv2_conv_Conv_output_0_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_2_m_0_cv2_conv_Conv_output_0_chain,
  NULL, &_model_2_m_0_cv2_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_m_0_cv1_conv_Conv_output_0_output, &_model_2_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Mul_output_0_layer, 13,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_2_m_0_cv1_act_Mul_output_0_chain,
  NULL, &_model_2_m_0_cv2_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv1_act_Sigmoid_output_0_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_2_m_0_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_2_m_0_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_m_0_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_2_m_0_cv1_conv_Conv_output_0_weights, &_model_2_m_0_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_m_0_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_2_m_0_cv1_conv_Conv_output_0_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_2_m_0_cv1_conv_Conv_output_0_chain,
  NULL, &_model_2_m_0_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_Split_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_Split_output_0_output0, &_model_2_Split_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_Split_output_0_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_Split_output_0_layer, 10,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &_model_2_Split_output_0_chain,
  NULL, &_model_2_m_0_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .outer_elems = 6400, 
  .outer_elems_stride = 128, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_cv1_conv_Conv_output_0_output, &_model_2_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv1_act_Mul_output_0_layer, 9,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_2_cv1_act_Mul_output_0_chain,
  NULL, &_model_2_Split_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv1_act_Sigmoid_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_2_cv1_act_Sigmoid_output_0_chain,
  NULL, &_model_2_cv1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_2_cv1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_2_cv1_conv_Conv_output_0_weights, &_model_2_cv1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_2_cv1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_2_cv1_conv_Conv_output_0_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_2_cv1_conv_Conv_output_0_chain,
  NULL, &_model_2_cv1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_1_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_1_conv_Conv_output_0_output, &_model_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_1_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_1_act_Mul_output_0_layer, 6,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_1_act_Mul_output_0_chain,
  NULL, &_model_2_cv1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_1_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_1_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_1_act_Sigmoid_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_1_act_Sigmoid_output_0_chain,
  NULL, &_model_1_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_1_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_1_conv_Conv_output_0_weights, &_model_1_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_1_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_1_conv_Conv_output_0_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_1_conv_Conv_output_0_chain,
  NULL, &_model_1_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_0_act_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_0_conv_Conv_output_0_output, &_model_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_0_act_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_0_act_Mul_output_0_layer, 3,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_0_act_Mul_output_0_chain,
  NULL, &_model_1_conv_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_0_act_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_0_act_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_0_act_Sigmoid_output_0_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_model_0_act_Sigmoid_output_0_chain,
  NULL, &_model_0_act_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &images_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_0_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_0_conv_Conv_output_0_weights, &_model_0_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_0_conv_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _model_0_conv_Conv_output_0_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_0_conv_Conv_output_0_chain,
  NULL, &_model_0_act_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  images_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &images_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &images_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  images_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &images_Transpose_chain,
  NULL, &_model_0_conv_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12066972, 1, 1),
    12066972, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4505600, 1, 1),
    4505600, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &images_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &output0_Transpose_0_output),
  &images_Transpose_layer, 0x7cfbc3a3, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12066972, 1, 1),
      12066972, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4505600, 1, 1),
      4505600, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &images_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &output0_Transpose_0_output),
  &images_Transpose_layer, 0x7cfbc3a3, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    images_output_array.data = AI_PTR(g_network_activations_map[0] + 2679168);
    images_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2679168);
    images_Transpose_output_array.data = AI_PTR(g_network_activations_map[0] + 1450368);
    images_Transpose_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1450368);
    _model_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2679168);
    _model_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2679168);
    _model_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_2_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_2_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 1024000);
    _model_2_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 1433600);
    _model_2_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 1433600);
    _model_2_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843776);
    _model_2_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843776);
    _model_2_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253376);
    _model_2_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253376);
    _model_2_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_2_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_2_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843776);
    _model_2_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843776);
    _model_2_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253376);
    _model_2_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253376);
    _model_2_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_2_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662976);
    _model_2_m_0_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_m_0_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_2_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_2_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_2_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_2_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_2_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_2_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_3_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_3_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_3_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1844352);
    _model_3_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1844352);
    _model_3_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_3_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_3_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_3_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_4_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_4_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1843456);
    _model_4_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1843456);
    _model_4_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253056);
    _model_4_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253056);
    _model_4_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662656);
    _model_4_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662656);
    _model_4_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_4_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 1843200);
    _model_4_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 2048000);
    _model_4_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 2048000);
    _model_4_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_4_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_4_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_4_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253952);
    _model_4_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_0_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_0_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_m_1_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_m_1_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_m_1_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_1_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_1_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_1_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_1_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2868352);
    _model_4_m_1_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2868352);
    _model_4_m_1_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_m_1_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_m_1_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_1_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2458752);
    _model_4_m_1_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_1_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2663552);
    _model_4_m_1_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2868352);
    _model_4_m_1_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2868352);
    _model_4_m_1_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_m_1_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_4_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_4_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_4_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_4_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_4_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_4_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_4_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_5_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_5_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_5_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2255104);
    _model_5_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2255104);
    _model_5_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_5_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_5_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_5_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_6_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_6_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_6_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2253312);
    _model_6_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2253312);
    _model_6_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_6_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_6_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_6_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 2252800);
    _model_6_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 2355200);
    _model_6_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 2355200);
    _model_6_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2459904);
    _model_6_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2459904);
    _model_6_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2459904);
    _model_6_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2459904);
    _model_6_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_0_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_0_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_m_1_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_1_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_1_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_1_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_1_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_1_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_m_1_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_1_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_6_m_1_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_m_1_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_6_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_6_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_6_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_6_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_6_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_7_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_7_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_7_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2462208);
    _model_7_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2462208);
    _model_7_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_7_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_7_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_7_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_8_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_8_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_8_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2458624);
    _model_8_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2458624);
    _model_8_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_8_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_8_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_8_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 2457600);
    _model_8_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 2508800);
    _model_8_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 2508800);
    _model_8_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2564608);
    _model_8_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2564608);
    _model_8_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_8_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_8_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2564608);
    _model_8_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2564608);
    _model_8_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_8_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_8_m_0_Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_m_0_Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3430400);
    _model_8_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3430400);
    _model_8_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_8_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_8_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_9_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2561024);
    _model_9_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2561024);
    _model_9_m_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_9_m_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_9_m_1_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_9_m_1_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3328000);
    _model_9_m_2_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_9_m_2_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3379200);
    _model_9_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3430400);
    _model_9_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3430400);
    _model_9_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_9_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_9_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_9_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_10_Resize_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_10_Resize_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_11_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_11_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_12_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_12_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_12_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3073536);
    _model_12_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3073536);
    _model_12_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3278336);
    _model_12_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3278336);
    _model_12_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3483136);
    _model_12_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3483136);
    _model_12_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_12_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 3072000);
    _model_12_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_12_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 3174400);
    _model_12_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3279104);
    _model_12_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3279104);
    _model_12_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3381504);
    _model_12_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3381504);
    _model_12_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3483904);
    _model_12_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3483904);
    _model_12_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3586304);
    _model_12_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3586304);
    _model_12_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_12_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3688704);
    _model_12_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3688704);
    _model_12_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3791104);
    _model_12_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3791104);
    _model_12_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1023232);
    _model_12_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1023232);
    _model_12_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3780864);
    _model_12_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3780864);
    _model_12_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_12_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_12_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 819200);
    _model_12_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 819200);
    _model_13_Resize_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_13_Resize_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _model_14_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_14_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_15_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _model_15_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 768);
    _model_15_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 768);
    _model_15_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_15_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _model_15_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 204800);
    _model_15_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 204800);
    _model_15_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 410752);
    _model_15_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 410752);
    _model_15_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2867200);
    _model_15_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2867200);
    _model_15_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 410752);
    _model_15_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 410752);
    _model_15_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2867200);
    _model_15_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2867200);
    _model_15_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_15_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_15_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_15_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv3_0_cv3_0_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv3_0_cv3_0_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409856);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409856);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 429056);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 429056);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv2_0_cv2_0_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_22_cv2_0_cv2_0_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 429056);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 429056);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3686400);
    _model_22_cv2_0_cv2_0_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_22_cv2_0_cv2_0_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4096000);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2662400);
    _model_16_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 429056);
    _model_16_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 429056);
    _model_16_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 431360);
    _model_16_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 431360);
    _model_16_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 533760);
    _model_16_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 533760);
    _model_16_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 636160);
    _model_16_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 636160);
    _model_17_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_17_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 429056);
    _model_18_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 429056);
    _model_18_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 429824);
    _model_18_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 429824);
    _model_18_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 634624);
    _model_18_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 634624);
    _model_18_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 429056);
    _model_18_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 429056);
    _model_18_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 531456);
    _model_18_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 531456);
    _model_18_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 636160);
    _model_18_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 636160);
    _model_18_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 738560);
    _model_18_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 738560);
    _model_18_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 840960);
    _model_18_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 840960);
    _model_18_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 636160);
    _model_18_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 636160);
    _model_18_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 738560);
    _model_18_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 738560);
    _model_18_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 840960);
    _model_18_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 840960);
    _model_18_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_18_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 634624);
    _model_18_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 634624);
    _model_18_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_18_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_18_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3481600);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 638464);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 638464);
    _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 740864);
    _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 740864);
    _model_22_cv3_1_cv3_1_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843264);
    _model_22_cv3_1_cv3_1_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843264);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 636160);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 636160);
    _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 738560);
    _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 738560);
    _model_22_cv3_1_cv3_1_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 840960);
    _model_22_cv3_1_cv3_1_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 840960);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 638656);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 638656);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 643264);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 643264);
    _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 745664);
    _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 745664);
    _model_22_cv2_1_cv2_1_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 848064);
    _model_22_cv2_1_cv2_1_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 848064);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 638656);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 638656);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 640960);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 640960);
    _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 743360);
    _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 743360);
    _model_22_cv2_1_cv2_1_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 845760);
    _model_22_cv2_1_cv2_1_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 845760);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 638656);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 638656);
    _model_19_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 741056);
    _model_19_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 741056);
    _model_19_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 745664);
    _model_19_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 745664);
    _model_19_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 796864);
    _model_19_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 796864);
    _model_19_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 848064);
    _model_19_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 848064);
    _model_20_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_20_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_21_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 741056);
    _model_21_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 741056);
    _model_21_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 742592);
    _model_21_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 742592);
    _model_21_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 844992);
    _model_21_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 844992);
    _model_21_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_21_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_21_Split_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 741056);
    _model_21_Split_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 741056);
    _model_21_Split_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 792256);
    _model_21_Split_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 792256);
    _model_21_m_0_cv1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_m_0_cv1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_m_0_cv1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 848064);
    _model_21_m_0_cv1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 848064);
    _model_21_m_0_cv1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 899264);
    _model_21_m_0_cv1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 899264);
    _model_21_m_0_cv1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 950464);
    _model_21_m_0_cv1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 950464);
    _model_21_m_0_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_m_0_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_m_0_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 848064);
    _model_21_m_0_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 848064);
    _model_21_m_0_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 899264);
    _model_21_m_0_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 899264);
    _model_21_m_0_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 950464);
    _model_21_m_0_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 950464);
    _model_21_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_21_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_21_cv2_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_cv2_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_21_cv2_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 844992);
    _model_21_cv2_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 844992);
    _model_21_cv2_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_21_cv2_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2560000);
    _model_21_cv2_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_21_cv2_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 852672);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 852672);
    _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 878272);
    _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 878272);
    _model_22_cv3_2_cv3_2_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 903872);
    _model_22_cv3_2_cv3_2_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 903872);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 845760);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 845760);
    _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 871360);
    _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 871360);
    _model_22_cv3_2_cv3_2_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 896960);
    _model_22_cv3_2_cv3_2_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 896960);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_Concat_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 844656);
    _model_22_Concat_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 844656);
    _model_22_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 869856);
    _model_22_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 869856);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 895056);
    _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 895056);
    _model_22_cv2_2_cv2_2_0_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 920656);
    _model_22_cv2_2_cv2_2_0_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 920656);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 895056);
    _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 895056);
    _model_22_cv2_2_cv2_2_1_act_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 920656);
    _model_22_cv2_2_cv2_2_1_act_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 920656);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    _model_22_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_dfl_Reshape_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 3814400);
    _model_22_dfl_Reshape_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3814400);
    _model_22_dfl_Transpose_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_dfl_Transpose_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3276800);
    _model_22_dfl_Softmax_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3814400);
    _model_22_dfl_Softmax_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3814400);
    _model_22_dfl_conv_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_dfl_conv_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_dfl_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_dfl_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_dfl_Reshape_1_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_dfl_Reshape_1_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_dfl_Reshape_1_output_0_to_chfirst_output_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_dfl_Reshape_1_output_0_to_chfirst_output_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_Slice_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Slice_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Slice_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Slice_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Add_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_Add_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Sub_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 650656);
    _model_22_Sub_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 650656);
    _model_22_Add_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Add_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Div_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Div_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 409600);
    _model_22_Concat_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Concat_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 667456);
    _model_22_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 633856);
    _model_22_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 633856);
    output0_output_array.data = AI_PTR(g_network_activations_map[0] + 667456);
    output0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 667456);
    output0_Transpose_0_output_array.data = AI_PTR(g_network_activations_map[0] + 843456);
    output0_Transpose_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 843456);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _model_22_Constant_14_output_0_3D_array.format |= AI_FMT_FLAG_CONST;
    _model_22_Constant_14_output_0_3D_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _model_22_Constant_14_output_0_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _model_22_Constant_15_output_0_3D_array.format |= AI_FMT_FLAG_CONST;
    _model_22_Constant_15_output_0_3D_array.data = AI_PTR(g_network_weights_map[0] + 4);
    _model_22_Constant_15_output_0_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    _model_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 8404);
    _model_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 8404);
    _model_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10132);
    _model_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10132);
    _model_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10196);
    _model_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10196);
    _model_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 28628);
    _model_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 28628);
    _model_2_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_2_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 28756);
    _model_2_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 28756);
    _model_2_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_2_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 32852);
    _model_2_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 32852);
    _model_2_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_2_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 32980);
    _model_2_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 32980);
    _model_2_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_2_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 32984);
    _model_2_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 32984);
    _model_2_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_2_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 42200);
    _model_2_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 42200);
    _model_2_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_2_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 42264);
    _model_2_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 42264);
    _model_2_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_2_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 51480);
    _model_2_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 51480);
    _model_2_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_2_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 51544);
    _model_2_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 51544);
    _model_2_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_2_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 57688);
    _model_2_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 57688);
    _model_3_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_3_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 57816);
    _model_3_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 57816);
    _model_3_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_3_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 131544);
    _model_3_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 131544);
    _model_4_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 131800);
    _model_4_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 131800);
    _model_4_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 148184);
    _model_4_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 148184);
    _model_4_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_4_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 148440);
    _model_4_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 148440);
    _model_4_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 148444);
    _model_4_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 148444);
    _model_4_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 185308);
    _model_4_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 185308);
    _model_4_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 185436);
    _model_4_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 185436);
    _model_4_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 222300);
    _model_4_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 222300);
    _model_4_m_1_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_1_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 222428);
    _model_4_m_1_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 222428);
    _model_4_m_1_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_1_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 259292);
    _model_4_m_1_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 259292);
    _model_4_m_1_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_1_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 259420);
    _model_4_m_1_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 259420);
    _model_4_m_1_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_m_1_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 296284);
    _model_4_m_1_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 296284);
    _model_4_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_4_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 296412);
    _model_4_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 296412);
    _model_4_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_4_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 329180);
    _model_4_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 329180);
    _model_5_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_5_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 329436);
    _model_5_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 329436);
    _model_5_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_5_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 624348);
    _model_5_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 624348);
    _model_6_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 624860);
    _model_6_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 624860);
    _model_6_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 690396);
    _model_6_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 690396);
    _model_6_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_6_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 690908);
    _model_6_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 690908);
    _model_6_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 690912);
    _model_6_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 690912);
    _model_6_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 838368);
    _model_6_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 838368);
    _model_6_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 838624);
    _model_6_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 838624);
    _model_6_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 986080);
    _model_6_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 986080);
    _model_6_m_1_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_1_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 986336);
    _model_6_m_1_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 986336);
    _model_6_m_1_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_1_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1133792);
    _model_6_m_1_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1133792);
    _model_6_m_1_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_1_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1134048);
    _model_6_m_1_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1134048);
    _model_6_m_1_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_m_1_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1281504);
    _model_6_m_1_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1281504);
    _model_6_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_6_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1281760);
    _model_6_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1281760);
    _model_6_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_6_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1412832);
    _model_6_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1412832);
    _model_7_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_7_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1413344);
    _model_7_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1413344);
    _model_7_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_7_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 2592992);
    _model_7_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2592992);
    _model_8_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_8_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2594016);
    _model_8_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2594016);
    _model_8_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_8_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 2856160);
    _model_8_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2856160);
    _model_8_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_8_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 2857184);
    _model_8_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 2857184);
    _model_8_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_8_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2857188);
    _model_8_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2857188);
    _model_8_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_8_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 3447012);
    _model_8_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3447012);
    _model_8_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_8_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 3447524);
    _model_8_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 3447524);
    _model_8_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_8_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4037348);
    _model_8_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4037348);
    _model_8_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_8_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 4037860);
    _model_8_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4037860);
    _model_8_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_8_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4431076);
    _model_8_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4431076);
    _model_9_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_9_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 4432100);
    _model_9_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4432100);
    _model_9_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_9_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4563172);
    _model_9_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4563172);
    _model_9_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_9_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 4563684);
    _model_9_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4563684);
    _model_9_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_9_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5087972);
    _model_9_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5087972);
    _model_12_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_12_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5088996);
    _model_12_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5088996);
    _model_12_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_12_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5285604);
    _model_12_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5285604);
    _model_12_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_12_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 5286116);
    _model_12_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 5286116);
    _model_12_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_12_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5286120);
    _model_12_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5286120);
    _model_12_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_12_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5433576);
    _model_12_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5433576);
    _model_12_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_12_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5433832);
    _model_12_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5433832);
    _model_12_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_12_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5581288);
    _model_12_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5581288);
    _model_12_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_12_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5581544);
    _model_12_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5581544);
    _model_12_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_12_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5679848);
    _model_12_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5679848);
    _model_15_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_15_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5680360);
    _model_15_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5680360);
    _model_15_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_15_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5729512);
    _model_15_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5729512);
    _model_15_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_15_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 5729768);
    _model_15_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 5729768);
    _model_15_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_15_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5729772);
    _model_15_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5729772);
    _model_15_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_15_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5766636);
    _model_15_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5766636);
    _model_15_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_15_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5766764);
    _model_15_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5766764);
    _model_15_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_15_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5803628);
    _model_15_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5803628);
    _model_15_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_15_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5803756);
    _model_15_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5803756);
    _model_15_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_15_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5828332);
    _model_15_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5828332);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5828588);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5828588);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5976044);
    _model_22_cv3_0_cv3_0_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5976044);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5976300);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5976300);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6123756);
    _model_22_cv3_0_cv3_0_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6123756);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6124012);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6124012);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_0_cv3_0_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6124780);
    _model_22_cv3_0_cv3_0_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6124780);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6124792);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6124792);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6272248);
    _model_22_cv2_0_cv2_0_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6272248);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6272504);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6272504);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6419960);
    _model_22_cv2_0_cv2_0_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6419960);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6420216);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6420216);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_0_cv2_0_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6436600);
    _model_22_cv2_0_cv2_0_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6436600);
    _model_16_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_16_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6436856);
    _model_16_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6436856);
    _model_16_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_16_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6584312);
    _model_16_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6584312);
    _model_18_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_18_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6584568);
    _model_18_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6584568);
    _model_18_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_18_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6682872);
    _model_18_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6682872);
    _model_18_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_18_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 6683384);
    _model_18_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 6683384);
    _model_18_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_18_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6683388);
    _model_18_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6683388);
    _model_18_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_18_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6830844);
    _model_18_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6830844);
    _model_18_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_18_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6831100);
    _model_18_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6831100);
    _model_18_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_18_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 6978556);
    _model_18_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6978556);
    _model_18_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_18_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 6978812);
    _model_18_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6978812);
    _model_18_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_18_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7077116);
    _model_18_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7077116);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7077628);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7077628);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7372540);
    _model_22_cv3_1_cv3_1_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7372540);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7372796);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7372796);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7520252);
    _model_22_cv3_1_cv3_1_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7520252);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7520508);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7520508);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_1_cv3_1_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7521276);
    _model_22_cv3_1_cv3_1_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7521276);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7521288);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7521288);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7816200);
    _model_22_cv2_1_cv2_1_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7816200);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7816456);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7816456);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7963912);
    _model_22_cv2_1_cv2_1_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7963912);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7964168);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7964168);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_1_cv2_1_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 7980552);
    _model_22_cv2_1_cv2_1_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 7980552);
    _model_19_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_19_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 7980808);
    _model_19_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 7980808);
    _model_19_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_19_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 8570632);
    _model_19_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8570632);
    _model_21_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_21_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 8571144);
    _model_21_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 8571144);
    _model_21_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_21_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 8964360);
    _model_21_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8964360);
    _model_21_Split_output_0_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    _model_21_Split_output_0_num_or_size_splits_array.data = AI_PTR(g_network_weights_map[0] + 8965384);
    _model_21_Split_output_0_num_or_size_splits_array.data_start = AI_PTR(g_network_weights_map[0] + 8965384);
    _model_21_m_0_cv1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_21_m_0_cv1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 8965388);
    _model_21_m_0_cv1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 8965388);
    _model_21_m_0_cv1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_21_m_0_cv1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 9555212);
    _model_21_m_0_cv1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 9555212);
    _model_21_m_0_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_21_m_0_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 9555724);
    _model_21_m_0_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 9555724);
    _model_21_m_0_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_21_m_0_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10145548);
    _model_21_m_0_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10145548);
    _model_21_cv2_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_21_cv2_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10146060);
    _model_21_cv2_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10146060);
    _model_21_cv2_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_21_cv2_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10539276);
    _model_21_cv2_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10539276);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10540300);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10540300);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 11130124);
    _model_22_cv3_2_cv3_2_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11130124);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 11130380);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11130380);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 11277836);
    _model_22_cv3_2_cv3_2_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11277836);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 11278092);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11278092);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv3_2_cv3_2_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 11278860);
    _model_22_cv3_2_cv3_2_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11278860);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 11278872);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11278872);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 11868696);
    _model_22_cv2_2_cv2_2_0_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11868696);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 11868952);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11868952);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 12016408);
    _model_22_cv2_2_cv2_2_1_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12016408);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 12016664);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 12016664);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_cv2_2_cv2_2_2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 12033048);
    _model_22_cv2_2_cv2_2_2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12033048);
    _model_22_dfl_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_22_dfl_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 12033304);
    _model_22_dfl_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 12033304);
    _model_22_dfl_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_22_dfl_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 12033368);
    _model_22_dfl_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12033368);
    _model_22_Constant_13_output_0_array.format |= AI_FMT_FLAG_CONST;
    _model_22_Constant_13_output_0_array.data = AI_PTR(g_network_weights_map[0] + 12033372);
    _model_22_Constant_13_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 12033372);
    _model_22_Constant_12_output_0_array.format |= AI_FMT_FLAG_CONST;
    _model_22_Constant_12_output_0_array.data = AI_PTR(g_network_weights_map[0] + 12050172);
    _model_22_Constant_12_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 12050172);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1051586352,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7cfbc3a3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1051586352,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7cfbc3a3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

