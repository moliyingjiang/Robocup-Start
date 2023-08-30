import paddle
import math
from x2paddle.op_mapper.pytorch2paddle import pytorch_custom_layer as x2paddle_nn

class DetectionModel(paddle.nn.Layer):
    def __init__(self):
        super(DetectionModel, self).__init__()
        self.conv2d0 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=12)
        self.silu0 = paddle.nn.Silu()
        self.conv2d1 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.silu1 = paddle.nn.Silu()
        self.conv2d2 = paddle.nn.Conv2D(out_channels=32, kernel_size=(1, 1), in_channels=64)
        self.silu2 = paddle.nn.Silu()
        self.conv2d3 = paddle.nn.Conv2D(out_channels=32, kernel_size=(1, 1), in_channels=32)
        self.silu3 = paddle.nn.Silu()
        self.conv2d4 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=32)
        self.silu4 = paddle.nn.Silu()
        self.conv2d5 = paddle.nn.Conv2D(out_channels=32, kernel_size=(1, 1), in_channels=64)
        self.silu5 = paddle.nn.Silu()
        self.conv2d6 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=64)
        self.silu6 = paddle.nn.Silu()
        self.conv2d7 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=128, kernel_size=(3, 3), in_channels=64)
        self.silu7 = paddle.nn.Silu()
        self.conv2d8 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=128)
        self.silu8 = paddle.nn.Silu()
        self.conv2d9 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=64)
        self.silu9 = paddle.nn.Silu()
        self.conv2d10 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.silu10 = paddle.nn.Silu()
        self.conv2d11 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=64)
        self.silu11 = paddle.nn.Silu()
        self.conv2d12 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.silu12 = paddle.nn.Silu()
        self.conv2d13 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=64)
        self.silu13 = paddle.nn.Silu()
        self.conv2d14 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.silu14 = paddle.nn.Silu()
        self.conv2d15 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=128)
        self.silu15 = paddle.nn.Silu()
        self.conv2d16 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu16 = paddle.nn.Silu()
        self.conv2d17 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=256, kernel_size=(3, 3), in_channels=128)
        self.silu17 = paddle.nn.Silu()
        self.conv2d18 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=256)
        self.silu18 = paddle.nn.Silu()
        self.conv2d19 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu19 = paddle.nn.Silu()
        self.conv2d20 = paddle.nn.Conv2D(padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu20 = paddle.nn.Silu()
        self.conv2d21 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu21 = paddle.nn.Silu()
        self.conv2d22 = paddle.nn.Conv2D(padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu22 = paddle.nn.Silu()
        self.conv2d23 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu23 = paddle.nn.Silu()
        self.conv2d24 = paddle.nn.Conv2D(padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu24 = paddle.nn.Silu()
        self.conv2d25 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=256)
        self.silu25 = paddle.nn.Silu()
        self.conv2d26 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=256)
        self.silu26 = paddle.nn.Silu()
        self.conv2d27 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=512, kernel_size=(3, 3), in_channels=256)
        self.silu27 = paddle.nn.Silu()
        self.conv2d28 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu28 = paddle.nn.Silu()
        self.pool2d0 = paddle.nn.MaxPool2D(kernel_size=[5, 5], stride=1, padding=2)
        self.pool2d1 = paddle.nn.MaxPool2D(kernel_size=[9, 9], stride=1, padding=4)
        self.pool2d2 = paddle.nn.MaxPool2D(kernel_size=[13, 13], stride=1, padding=6)
        self.conv2d29 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), in_channels=1024)
        self.silu29 = paddle.nn.Silu()
        self.conv2d30 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu30 = paddle.nn.Silu()
        self.conv2d31 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=256)
        self.silu31 = paddle.nn.Silu()
        self.conv2d32 = paddle.nn.Conv2D(padding=1, out_channels=256, kernel_size=(3, 3), in_channels=256)
        self.silu32 = paddle.nn.Silu()
        self.conv2d33 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu33 = paddle.nn.Silu()
        self.conv2d34 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), in_channels=512)
        self.silu34 = paddle.nn.Silu()
        self.conv2d35 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu35 = paddle.nn.Silu()
        self.conv2d36 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=512)
        self.silu36 = paddle.nn.Silu()
        self.conv2d37 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu37 = paddle.nn.Silu()
        self.conv2d38 = paddle.nn.Conv2D(padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu38 = paddle.nn.Silu()
        self.conv2d39 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=512)
        self.silu39 = paddle.nn.Silu()
        self.conv2d40 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=256)
        self.silu40 = paddle.nn.Silu()
        self.conv2d41 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=256)
        self.silu41 = paddle.nn.Silu()
        self.conv2d42 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=256)
        self.silu42 = paddle.nn.Silu()
        self.conv2d43 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=64)
        self.silu43 = paddle.nn.Silu()
        self.conv2d44 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.silu44 = paddle.nn.Silu()
        self.conv2d45 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=256)
        self.silu45 = paddle.nn.Silu()
        self.conv2d46 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu46 = paddle.nn.Silu()
        self.conv2d47 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu47 = paddle.nn.Silu()
        self.conv2d48 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=256)
        self.silu48 = paddle.nn.Silu()
        self.conv2d49 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.silu49 = paddle.nn.Silu()
        self.conv2d50 = paddle.nn.Conv2D(padding=1, out_channels=128, kernel_size=(3, 3), in_channels=128)
        self.silu50 = paddle.nn.Silu()
        self.conv2d51 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=256)
        self.silu51 = paddle.nn.Silu()
        self.conv2d52 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=256)
        self.silu52 = paddle.nn.Silu()
        self.conv2d53 = paddle.nn.Conv2D(stride=2, padding=1, out_channels=256, kernel_size=(3, 3), in_channels=256)
        self.silu53 = paddle.nn.Silu()
        self.conv2d54 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu54 = paddle.nn.Silu()
        self.conv2d55 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=256)
        self.silu55 = paddle.nn.Silu()
        self.conv2d56 = paddle.nn.Conv2D(padding=1, out_channels=256, kernel_size=(3, 3), in_channels=256)
        self.silu56 = paddle.nn.Silu()
        self.conv2d57 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=512)
        self.silu57 = paddle.nn.Silu()
        self.conv2d58 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), in_channels=512)
        self.silu58 = paddle.nn.Silu()
        self.x767 = self.create_parameter(dtype='float32', shape=(1, 3, 20, 20, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x768 = self.create_parameter(dtype='float32', shape=(1, 3, 20, 20, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x769 = self.create_parameter(dtype='float32', shape=(1, 3, 40, 40, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x770 = self.create_parameter(dtype='float32', shape=(1, 3, 40, 40, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x771 = self.create_parameter(dtype='float32', shape=(1, 3, 80, 80, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x775 = self.create_parameter(dtype='float32', shape=(1, 3, 80, 80, 2), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.conv2d59 = paddle.nn.Conv2D(out_channels=255, kernel_size=(1, 1), in_channels=128)
        self.sigmoid0 = paddle.nn.Sigmoid()
        self.conv2d60 = paddle.nn.Conv2D(out_channels=255, kernel_size=(1, 1), in_channels=256)
        self.sigmoid1 = paddle.nn.Sigmoid()
        self.conv2d61 = paddle.nn.Conv2D(out_channels=255, kernel_size=(1, 1), in_channels=512)
        self.sigmoid2 = paddle.nn.Sigmoid()

    def forward(self, x0):
        x44_list = [0]
        x43_list = [2147483647]
        x45_list = [2]
        x47 = paddle.strided_slice(x=x0, axes=x45_list, starts=x44_list, ends=x43_list, strides=x45_list)
        x42_list = [3]
        x44_list = [0]
        x43_list = [2147483647]
        x45_list = [2]
        x48 = paddle.strided_slice(x=x47, axes=x42_list, starts=x44_list, ends=x43_list, strides=x45_list)
        x41_list = [1]
        x43_list = [2147483647]
        x45_list = [2]
        x49 = paddle.strided_slice(x=x0, axes=x45_list, starts=x41_list, ends=x43_list, strides=x45_list)
        x42_list = [3]
        x44_list = [0]
        x43_list = [2147483647]
        x45_list = [2]
        x50 = paddle.strided_slice(x=x49, axes=x42_list, starts=x44_list, ends=x43_list, strides=x45_list)
        x44_list = [0]
        x43_list = [2147483647]
        x45_list = [2]
        x51 = paddle.strided_slice(x=x0, axes=x45_list, starts=x44_list, ends=x43_list, strides=x45_list)
        x42_list = [3]
        x41_list = [1]
        x43_list = [2147483647]
        x45_list = [2]
        x52 = paddle.strided_slice(x=x51, axes=x42_list, starts=x41_list, ends=x43_list, strides=x45_list)
        x41_list = [1]
        x43_list = [2147483647]
        x45_list = [2]
        x53 = paddle.strided_slice(x=x0, axes=x45_list, starts=x41_list, ends=x43_list, strides=x45_list)
        x42_list = [3]
        x41_list = [1]
        x43_list = [2147483647]
        x45_list = [2]
        x54 = paddle.strided_slice(x=x53, axes=x42_list, starts=x41_list, ends=x43_list, strides=x45_list)
        x55 = [x48, x50, x52, x54]
        x56 = paddle.concat(x=x55, axis=1)
        x64 = self.conv2d0(x56)
        x65 = self.silu0(x64)
        x76 = self.conv2d1(x65)
        x77 = self.silu1(x76)
        x92 = self.conv2d2(x77)
        x93 = self.silu2(x92)
        x104 = self.conv2d3(x93)
        x105 = self.silu3(x104)
        x113 = self.conv2d4(x105)
        x114 = self.silu4(x113)
        x115 = x93 + x114
        x123 = self.conv2d5(x77)
        x124 = self.silu5(x123)
        x125 = [x115, x124]
        x126 = paddle.concat(x=x125, axis=1)
        x134 = self.conv2d6(x126)
        x135 = self.silu6(x134)
        x146 = self.conv2d7(x135)
        x147 = self.silu7(x146)
        x162 = self.conv2d8(x147)
        x163 = self.silu8(x162)
        x176 = self.conv2d9(x163)
        x177 = self.silu9(x176)
        x185 = self.conv2d10(x177)
        x186 = self.silu10(x185)
        x187 = x163 + x186
        x197 = self.conv2d11(x187)
        x198 = self.silu11(x197)
        x206 = self.conv2d12(x198)
        x207 = self.silu12(x206)
        x208 = x187 + x207
        x218 = self.conv2d13(x208)
        x219 = self.silu13(x218)
        x227 = self.conv2d14(x219)
        x228 = self.silu14(x227)
        x229 = x208 + x228
        x237 = self.conv2d15(x147)
        x238 = self.silu15(x237)
        x239 = [x229, x238]
        x240 = paddle.concat(x=x239, axis=1)
        x248 = self.conv2d16(x240)
        x249 = self.silu16(x248)
        x260 = self.conv2d17(x249)
        x261 = self.silu17(x260)
        x276 = self.conv2d18(x261)
        x277 = self.silu18(x276)
        x290 = self.conv2d19(x277)
        x291 = self.silu19(x290)
        x299 = self.conv2d20(x291)
        x300 = self.silu20(x299)
        x301 = x277 + x300
        x311 = self.conv2d21(x301)
        x312 = self.silu21(x311)
        x320 = self.conv2d22(x312)
        x321 = self.silu22(x320)
        x322 = x301 + x321
        x332 = self.conv2d23(x322)
        x333 = self.silu23(x332)
        x341 = self.conv2d24(x333)
        x342 = self.silu24(x341)
        x343 = x322 + x342
        x351 = self.conv2d25(x261)
        x352 = self.silu25(x351)
        x353 = [x343, x352]
        x354 = paddle.concat(x=x353, axis=1)
        x362 = self.conv2d26(x354)
        x363 = self.silu26(x362)
        x374 = self.conv2d27(x363)
        x375 = self.silu27(x374)
        x388 = self.conv2d28(x375)
        x389 = self.silu28(x388)
        x394 = self.pool2d0(x389)
        x399 = self.pool2d1(x389)
        x404 = self.pool2d2(x389)
        x405 = [x389, x394, x399, x404]
        x406 = paddle.concat(x=x405, axis=1)
        x414 = self.conv2d29(x406)
        x415 = self.silu29(x414)
        x430 = self.conv2d30(x415)
        x431 = self.silu30(x430)
        x442 = self.conv2d31(x431)
        x443 = self.silu31(x442)
        x451 = self.conv2d32(x443)
        x452 = self.silu32(x451)
        x460 = self.conv2d33(x415)
        x461 = self.silu33(x460)
        x462 = [x452, x461]
        x463 = paddle.concat(x=x462, axis=1)
        x471 = self.conv2d34(x463)
        x472 = self.silu34(x471)
        x483 = self.conv2d35(x472)
        x484 = self.silu35(x483)
        x486 = [2.0, 2.0]
        x487 = paddle.nn.functional.interpolate(x=x484, scale_factor=x486, mode='nearest')
        x489 = [x487, x363]
        x490 = paddle.concat(x=x489, axis=1)
        x505 = self.conv2d36(x490)
        x506 = self.silu36(x505)
        x517 = self.conv2d37(x506)
        x518 = self.silu37(x517)
        x526 = self.conv2d38(x518)
        x527 = self.silu38(x526)
        x535 = self.conv2d39(x490)
        x536 = self.silu39(x535)
        x537 = [x527, x536]
        x538 = paddle.concat(x=x537, axis=1)
        x546 = self.conv2d40(x538)
        x547 = self.silu40(x546)
        x558 = self.conv2d41(x547)
        x559 = self.silu41(x558)
        x561 = [2.0, 2.0]
        x562 = paddle.nn.functional.interpolate(x=x559, scale_factor=x561, mode='nearest')
        x564 = [x562, x249]
        x565 = paddle.concat(x=x564, axis=1)
        x580 = self.conv2d42(x565)
        x581 = self.silu42(x580)
        x592 = self.conv2d43(x581)
        x593 = self.silu43(x592)
        x601 = self.conv2d44(x593)
        x602 = self.silu44(x601)
        x610 = self.conv2d45(x565)
        x611 = self.silu45(x610)
        x612 = [x602, x611]
        x613 = paddle.concat(x=x612, axis=1)
        x621 = self.conv2d46(x613)
        x622 = self.silu46(x621)
        x633 = self.conv2d47(x622)
        x634 = self.silu47(x633)
        x636 = [x634, x559]
        x637 = paddle.concat(x=x636, axis=1)
        x652 = self.conv2d48(x637)
        x653 = self.silu48(x652)
        x664 = self.conv2d49(x653)
        x665 = self.silu49(x664)
        x673 = self.conv2d50(x665)
        x674 = self.silu50(x673)
        x682 = self.conv2d51(x637)
        x683 = self.silu51(x682)
        x684 = [x674, x683]
        x685 = paddle.concat(x=x684, axis=1)
        x693 = self.conv2d52(x685)
        x694 = self.silu52(x693)
        x705 = self.conv2d53(x694)
        x706 = self.silu53(x705)
        x708 = [x706, x484]
        x709 = paddle.concat(x=x708, axis=1)
        x724 = self.conv2d54(x709)
        x725 = self.silu54(x724)
        x736 = self.conv2d55(x725)
        x737 = self.silu55(x736)
        x745 = self.conv2d56(x737)
        x746 = self.silu56(x745)
        x754 = self.conv2d57(x709)
        x755 = self.silu57(x754)
        x756 = [x746, x755]
        x757 = paddle.concat(x=x756, axis=1)
        x765 = self.conv2d58(x757)
        x766 = self.silu58(x765)
        x767 = self.x767
        x768 = self.x768
        x769 = self.x769
        x770 = self.x770
        x771 = self.x771
        x772 = 2
        x775 = self.x775
        x794 = self.conv2d59(x622)
        x796 = paddle.reshape(x=x794, shape=[1, 3, 85, 80, 80])
        x798 = paddle.transpose(x=x796, perm=[0, 1, 3, 4, 2])
        x799 = x798
        x776_list = [4]
        x774_list = [5]
        x773_list = [2147483647]
        x778_list = [1]
        x800 = paddle.strided_slice(x=x799, axes=x776_list, starts=x774_list, ends=x773_list, strides=x778_list)
        x801 = self.sigmoid0(x799)
        x803 = paddle.split(x=x801, num_or_sections=[2, 2, 81], axis=4)
        x804, x805, x806 = x803
        x807 = x804 * x772
        x808 = x807 + x775
        x809 = 8.0
        x810 = x808 * x809
        x811 = x805 * x772
        x812 = paddle.pow(x=x811, y=2)
        x813 = x812 * x771
        x814 = [x810, x813, x806]
        x815 = paddle.concat(x=x814, axis=4)
        x817 = paddle.reshape(x=x815, shape=[1, 19200, 85])
        x819 = paddle.reshape(x=x800, shape=[1, -1, 80])
        x826 = self.conv2d60(x694)
        x828 = paddle.reshape(x=x826, shape=[1, 3, 85, 40, 40])
        x830 = paddle.transpose(x=x828, perm=[0, 1, 3, 4, 2])
        x831 = x830
        x776_list = [4]
        x774_list = [5]
        x773_list = [2147483647]
        x778_list = [1]
        x832 = paddle.strided_slice(x=x831, axes=x776_list, starts=x774_list, ends=x773_list, strides=x778_list)
        x833 = self.sigmoid1(x831)
        x835 = paddle.split(x=x833, num_or_sections=[2, 2, 81], axis=4)
        x836, x837, x838 = x835
        x839 = x836 * x772
        x840 = x839 + x770
        x841 = 16.0
        x842 = x840 * x841
        x843 = x837 * x772
        x844 = paddle.pow(x=x843, y=2)
        x845 = x844 * x769
        x846 = [x842, x845, x838]
        x847 = paddle.concat(x=x846, axis=4)
        x849 = paddle.reshape(x=x847, shape=[1, 4800, 85])
        x851 = paddle.reshape(x=x832, shape=[1, -1, 80])
        x858 = self.conv2d61(x766)
        x860 = paddle.reshape(x=x858, shape=[1, 3, 85, 20, 20])
        x862 = paddle.transpose(x=x860, perm=[0, 1, 3, 4, 2])
        x863 = x862
        x776_list = [4]
        x774_list = [5]
        x773_list = [2147483647]
        x778_list = [1]
        x864 = paddle.strided_slice(x=x863, axes=x776_list, starts=x774_list, ends=x773_list, strides=x778_list)
        x865 = self.sigmoid2(x863)
        x867 = paddle.split(x=x865, num_or_sections=[2, 2, 81], axis=4)
        x868, x869, x870 = x867
        x871 = x868 * x772
        x872 = x871 + x768
        x873 = 32.0
        x874 = x872 * x873
        x875 = x869 * x772
        x876 = paddle.pow(x=x875, y=2)
        x877 = x876 * x767
        x878 = [x874, x877, x870]
        x879 = paddle.concat(x=x878, axis=4)
        x881 = paddle.reshape(x=x879, shape=[1, 1200, 85])
        x883 = paddle.reshape(x=x864, shape=[1, -1, 80])
        x884 = [x817, x849, x881]
        x885 = paddle.concat(x=x884, axis=1)
        x886 = [x819, x851, x883]
        x887 = paddle.concat(x=x886, axis=1)
        x888 = (x799, x831, x863, x885, x887)
        x889, x890, x891, x892, x893 = x888
        x894 = [x889, x890, x891]
        x895 = (x892, x893, x894)
        return x895

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 640, 640], type-float32.
    paddle.disable_static()
    params = paddle.load(r'E:\jiehe\models\yolov5s_paddle_model\model.pdparams')
    model = DetectionModel()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    out = model(x0)
    return out
