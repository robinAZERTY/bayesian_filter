/*
Not tested yet !


To test the basic functionalities of the EKF, here is an example of a simple one-dimensional problem.
Temperature estimation in a controlled environment (such as an oven with a heating element).
The scenario is as follows:
You have an oven, and you want to precisely control its temperature using a heating element.
Let's assume you only have a temperature sensor that is highly imprecise, with significant noise and a low sampling rate.
You also have a physical model of the oven, allowing you to predict the temperature based on the heating power.

The constants characterizing the oven are as follows:
- Thermal inertia: the oven heats up and cools down slowly.
- Thermal losses: the temperature inside the oven tends to equalize naturally with the external temperature.
- External temperature: the external temperature is a known value.

x: oven temperature
u: heating power
c: oven parameters (time step, thermal inertia, thermal losses, external temperature)

The oven temperature evolves according to the following formula:
f(x, u, c) = x + u * c0 / c1 - c0 * (x - c3) / c2

The measurement function is simply the temperature measured by the sensor:
h(x, c) = x

Note: An Extended Kalman Filter (EKF) may not be the ideal solution for this problem since it is linear,
but it should be a good way to check that the EKF is functioning correctly for simple problems.
*/

#include <ekf.hpp>
#ifdef NATIVE
#include <iostream>
#define NAN std::nan("")
#define isnan(x) std::isnan(x)
#include <cmath>
#endif


// Note : In ekf, c[0] is always the time step, c[1:] are the remaining parameters

template <typename T>
internal::tmp<Vector<T>> && f(const Vector<T> &x, const Vector<T> &u, const Vector<T> &c)
{
    auto *ret = internal::tmp<Vector<T>>::get(x.size()); // ask for a temporary variable of the same size as x
    (*ret)[0] = x[0] + c[0] *(u[0]/c[1] - (x[0]-c[3])/c[2]); // compute the new temperature
    return internal::move(*ret); // return the temporary variable
}

// simple measurement function
template <typename T>
internal::tmp<Vector<T>> && h(const Vector<T> &x, const Vector<T> &c)
{
    auto *ret = internal::tmp<Vector<T>>::get(1); // ask for a temporary variable of size 2
    (*ret)[0] = x[0];
    return internal::move(*ret);
}


Vector<double> Z(1); // measurement
symMatrix<double> R(1, 1); // measurement noise

// simulation of commands and measurements
double UU[] = { -808.391749, -427.135143, 1794.994554, 920.280779, 1075.273081, 1168.853995, 1394.548759, 478.773480, 1977.432822, 987.228675, 1580.633611, 1864.301185, 1940.374073, 707.840166, 908.051176, 1168.576696, 1333.565602, 1831.359725, 1445.702961, 2438.562266, 1001.836307, 1467.214423, 1286.357278, 1167.833684, 1208.114000, 1857.909790, 2399.380194, 1527.520057, 927.380818, 2037.182433, 863.511995, 1633.681665, 1056.971556, 523.515525, 1723.401814, 1734.212943, 1248.567040, 847.546207, 1287.231252, 570.272094, 2332.495559, 1598.555899, 1700.378230, 1403.193687, 1692.663330, 1966.132034, 1853.681351, 1705.220473, 2332.139474, 1930.308872, 2753.166405, 2584.194850, 2499.965849, 2126.583636, 2743.253373, 2626.972434, 3088.895959, 2148.304959, 1937.371776, 1803.843287, 1816.703157, 1653.653791, 2047.270866, 2888.919337, 2037.291960, 1928.159913, 2591.204473, 2905.873932, 2075.640898, 2330.324537, 2328.110168, 2835.009814, 2543.046091, 3235.266791, 3012.314745, 1776.061634, 2926.008012, 2313.613491, 1918.304785, 2195.281552, 2807.562108, 2035.491274, 2151.734542, 2681.656687, 2248.291517, 3000.251808, 1714.987778, 2109.465521, 4055.496356, 3035.420677, 3356.436960, 3356.575880, 2755.429058, 2911.201443, 3249.484039, 3948.881979, 3058.784422, 3477.727154, 3396.747847, 3091.205560, 3827.181643, 3632.440879, 2340.470700, 3454.613072, 2822.084571, 3743.386872, 2258.881111, 3163.446789, 2758.284983, 4153.388847, 3660.687699, 3457.239278, 2163.850375, 2934.751430, 3062.987220, 2794.285160, 3304.889728, 2290.586818, 2900.424600, 3183.367610, 2931.746678, 3773.674658, 3648.658987, 3910.889490, 3494.874964, 3297.985005, 3907.225471, 4033.614785, 3443.927582, 3714.731772, 3820.907543, 3542.111139, 3065.745538, 2839.750757, 2889.365462, 3038.090533, 3683.713231, 3981.796898, 3338.110099, 4102.091661, 3923.925244, 4591.692276, 4608.000123, 3717.335644, 3902.527945, 3102.180805, 3155.595455, 3430.098108, 3500.457020, 4686.581918, 4736.522866, 4735.426999, 4375.947780, 3739.407656, 4144.101087, 4440.045863, 3732.801155, 4417.852033, 3736.497161, 4196.051227, 4179.548832, 4571.408798, 4166.744386, 3616.784853, 5202.251515, 5106.520640, 4336.500215, 4191.012165, 4196.575647, 4261.941754, 4784.712799, 4037.861829, 4022.567280, 5092.774428, 5039.041308, 4207.803377, 4333.350139, 3820.030001, 4681.455924, 4214.220782, 5182.627310, 4791.770895, 4423.709027, 4177.641562, 4351.748293, 5410.985531, 5232.389026, 5672.997363, 4489.014363, 4646.207697, 5513.725741, 5243.078415, 4683.101360, 5294.508799, 4889.771517, 5180.495105, 5006.227032, 4136.382613, 4675.968892, 4759.538916, 5251.517425, 5572.000101, 5183.895439, 5225.342695, 5245.403804, 4438.457736, 6177.995868, 5591.857662, 5953.684202, 5785.652251, 5077.685765, 5604.413610, 6016.583681, 4392.360697, 4223.675120, 4722.430542, 5204.394594, 5167.633648, 5321.095758, 6351.944332, 5641.742919, 5475.136056, 5758.858645, 4720.508477, 5127.198988, 6121.029701, 5029.852338, 5944.582827, 5149.471419, 5448.866107, 5616.326480, 5545.299132, 5395.894035, 6472.198655, 5991.971817, 5487.579997, 5717.317252, 6533.019736, 5454.321634, 5942.873739, 5769.586189, 6545.635233, 5636.061628, 6533.116483, 5603.885658, 6240.878193, 7017.433608, 5475.037967, 6813.273118, 5991.290400, 6514.930420, 6265.100068, 5591.815257, 5248.728651, 5694.423412, 5819.320964, 5709.863852, 6178.295977, 6037.626540, 6402.509820, 6721.702260, 6784.423595, 5777.463864, 6507.305263, 6625.558028, 5827.334422, 5693.890110, 6174.720765, 5975.038744, 6566.347967, 6354.643380, 6491.567369, 5811.886390, 6337.396795, 6588.526684, 7180.606778, 6780.072470, 7049.476486, 6518.957152, 6951.365611, 6967.801868, 7036.709485, 6359.811077, 6583.935788, 6217.435836, 6661.450984, 6326.508979, 6887.221091, 6734.948285, 7577.377005, 6314.098276, 7591.146569, 6775.748444, 6473.752923, 5875.503538, 6720.436125, 6251.416176, 8241.934851, 5516.949874, 7630.117183, 5196.446494, 5025.105000, 5081.106602, 5964.004981, 5215.878105, 5446.225715, 4603.786189, 4307.974253, 4213.655021, 5597.955201, 6085.444856, 4834.953321, 4992.890421, 6174.938285, 4577.174918, 5275.195268, 5019.180212, 4977.457009, 4550.793502, 4959.751142, 5637.966519, 4907.592319, 4581.363408, 5120.310224, 5620.680544, 5244.464238, 4245.011145, 4444.083742, 5138.031761, 5356.346163, 5995.816798, 5130.271250, 4537.106397, 4531.883190, 4473.292063, 5389.991349, 4866.548609, 3988.334670, 4874.734544, 5445.737875, 5192.584977, 3880.329262, 4784.193278, 5788.889476, 6621.243273, 4715.798279, 4966.480293, 5064.167035, 5381.847467, 4099.932575, 5186.845524, 4736.549176, 4957.819947, 4850.550843, 4012.951757, 5024.284022, 5517.354904, 5541.184379, 4481.235621, 4703.459827, 4893.659153, 4927.996775, 4931.524742, 4816.215831, 4928.087901, 4384.887230, 4951.756008, 5279.397221, 5043.178865, 5664.520983, 4837.246021, 5753.976118, 5517.258963, 5208.759380, 4965.579747, 4439.262817, 5666.749086, 3975.718220, 4967.562883, 5157.334460, 5087.445953, 5241.700026, 3958.245598, 5361.671299, 4928.225939, 4726.062145, 4675.312399, 6078.758279, 5740.051009, 5378.129143, 4136.250802, 4858.481192, 5303.714242, 5374.239928, 5217.864997, 5171.929564, 4832.743437, 5081.153242, 4050.626658, 4937.997377, 5069.588047, 4409.623372, 5832.514938, 4808.990078, 4634.645673, 4998.594251, 4768.036878, 5517.754943, 5240.285688, 6011.365373, 5050.040993, 4654.269922, 5220.461594, 5464.816421, 4967.550467, 5145.742864, 4835.230605, 4546.750574, 6076.607281, 5981.458017, 4992.164988, 5856.274190, 5043.720272, 4694.899823, 4025.769964, 3794.530833, 4962.555758, 4978.071570, 5121.247197, 5696.648985, 4835.929218, 5225.872520, 5042.472296, 4747.732227, 4612.318733, 5264.639063, 4213.948115, 5837.610010, 5174.937418, 5442.387817, 5387.766503, 4641.474753, 5165.819157, 5100.567779, 5463.896719, 5435.591661, 5052.889334, 6020.767241, 5725.250416, 5750.839977, 5410.651032, 4644.446916, 5433.677510, 4345.652100, 5055.485955, 4849.797889, 5351.730797, 5925.569234, 4622.784114, 5352.908746, 5019.062613, 4543.398917, 5218.072767, 5289.206574, 5318.524402, 4594.949243, 4705.756122, 4594.030751, 4517.324207, 4933.759755, 4880.728726, 4809.934423, 4456.451767, 4686.649717, 4799.437122, 5538.265432, 5265.684957, 4759.388938, 5103.482048, 5529.400779, 6090.971940, 5017.563758, 5324.258349, 5017.111110, 5222.347186, 4307.886689, 4952.596253, 5290.294171, 4615.001518, 5625.095947, 4385.521608, 5371.146136, 4315.295940, 4575.215428, 4681.396088, 4797.769950, 3971.674025, 4167.303727, 5082.530756, 4480.315164 };
double ZZ[] = { NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 22.127256, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 14.735154, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 27.682002, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 39.516015, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 28.847134, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 21.240553, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 33.767847, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 27.123182, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 38.181910, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 27.481835, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 45.136615, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 31.107731, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 34.115666, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 47.278962, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 34.307610, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 49.299423, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 55.145620, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 37.851878, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 58.435052, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 44.631667, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 67.858608, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 69.521151, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 52.964697, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 63.038940, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 72.095440, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 70.918874, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 67.155274, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 74.708538, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 75.877301, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 82.630868, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 70.373810, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 72.356667, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 59.356017, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 75.403988, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 68.936340, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 54.715367, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 67.054043, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 71.542513, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 61.819123, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 58.368590, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 78.084416, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 69.922438, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 67.850835, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 67.395611, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 73.107723, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 69.571203, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 73.282385, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 62.865790, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 84.796754, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN };


Ekf<1, 1, 4> ekf; // 1 state, 1 command, 4 parameters


void setup() {
    // set the filter
    ekf.X[0] = 0.0; // initial estimated temperature
    ekf.P(0, 0) = 40000.0; // initial covariance (high uncertainty)
    ekf.Cov_U(0, 0) = 250000.0; // covariance of the command

    ekf.setPredictionFunction(f);
    ekf.C[0] = 1; // time step
    ekf.C[1] = 10000; // thermal inertia
    ekf.C[2] = 100;  // thermal losses
    ekf.C[3] = 20; // external temperature

    ekf.setMeasurementFunction(h,1);
    R(0, 0) = 62.5; // measurement covariance

}

unsigned int i = 0;
void loop() {

    ekf.U[0] = UU[i]; // set the command
    ekf.predict(); // predict the temperature using the command
    if (!isnan(ZZ[i])) // if a measurement is available
    {
        Z[0] = ZZ[i]; // set the measurement
        ekf.update(Z, R); // update the temperature estimate using the measurement
    }
    // EKF estimates are in ekf.X
    if i >= sizeof(UU)/sizeof(UU[0])
        exit(0);
}

#ifdef NATIVE
int main(int argc, char **argv) {
    setup();
    while (true)
        loop();
    return 0;
}
#endif
