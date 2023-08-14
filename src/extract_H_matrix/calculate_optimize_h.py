import math

import numpy as np
from scipy.optimize import minimize, least_squares

def compute_error1(H, A, B, X, Y, fu, fv, v0, u0):
    X = np.array(X)
    Y = np.array(Y)
    u = ((fu * H[0] + u0 * H[4]) * X + (fu * H[1] + u0 * H[5]) * Y + fu * H[6] + u0 * H[8]) / (H[4] * X + H[5] * Y + H[8])
    v = ((fv * H[2] + v0 * H[4]) * X + (fv * H[3] + v0 * H[5]) * Y + fv * H[7] + v0 * H[8]) / (H[4] * X + H[5] * Y + H[8])
    error = np.sum((A * u + B * v - 1)**2)
    return error


def compute_error2(H, A, B, X, Y, fu, fv, v0, u0):
    X = np.array(X)
    Y = np.array(Y)
    u = ((fu * H[0] + u0 * H[4]) * X + (fu * H[1] + u0 * H[5]) * Y + fu * H[6] + u0 * H[8]) / (
                H[4] * X + H[5] * Y + H[8])
    v = ((fv * H[2] + v0 * H[4]) * X + (fv * H[3] + v0 * H[5]) * Y + fv * H[7] + v0 * H[8]) / (
                H[4] * X + H[5] * Y + H[8])

    A = np.array(A)  # Convert A to a NumPy array
    B = np.array(B)  # Convert B to a NumPy array

    error = np.sum(np.abs(A * u + B * v - 1)) / math.sqrt(
        np.sum(A ** 2 + B ** 2))  # Use np.abs() for element-wise absolute value
    return error

def compute_H(A, B, X, Y, fu, fv, u0, v0):
    k = np.zeros((9, 9))
    for i in range(9):
        k[i, :] = [A[i] * fu * X[i], A[i] * fu * Y[i], B[i] * fv * X[i], B[i] * fv * Y[i],
                   (A[i] * u0 + B[i] * v0 - 1) * X[i], (A[i] * u0 + B[i] * v0 - 1) * Y[i],
                   A[i] * fu, B[i] * fv, (A[i] * u0 + B[i] * v0 - 1)]

    r = 0.000001 + 0.000005 * np.random.rand(9, 1)
    print(r)

    k_inverse = np.linalg.inv(k)
    H = np.dot(k_inverse, r)

    H_matrix = H.reshape(3, 3)
    return H_matrix


def optimizeH(H0, A, B, X, Y, fu, fv, v0, u0):
    optimization_function = lambda H: compute_error1(H, A, B, X, Y, fu, fv, v0, u0)
    H = minimize(optimization_function, H0).x
    return H

def Hoptimizationlsqnonlin(H0, A, B, X, Y, fu, fv, v0, u0):
    optimization_function = lambda H: compute_error2(H, A, B, X, Y, fu, fv, v0, u0)
    options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 1000}
    H = least_squares(optimization_function, H0, method='trf', xtol=options['xtol'], ftol=options['ftol'], max_nfev=options['maxfev']).x
    return H

if __name__ == '__main__':
    A = [0.004878048780487805, 0.0049200492004954684, 0.004405286343612335, 0.0061654392875546745, 0.00546448087431694
        , 0.005574136008924396, 0.006896551724137931, 0.0064516129032258064, 0.0063278623690008985, 0.009708737864077669
        , 0.009193590753893328, 0.008302808302823211, 0.008, 0.007194244604316547, 0.006899195093913486
        , 0.006283320639763255, 0.005771248688357935, 0.005412929167959454, 0.005241875093611759, 0.004916136495089816
        , 0.0046035805626643, 0.004524886877828055, 0.00442477876106403, 0.004461371055499773, 0.0047753418710675824
        , 0.004592545124429143, 0.004302581548932063, 0.006266666666673594, 0.005450941526266295, 0.005290383258880654
        , 0.008283397955595186, 0.0069247717972984865, 0.006522972206474341, 0.012048192771084338, 0.010238148230596803
        , 0.00875273522977036, 0.024390243902439025, 0.01815592737636265, 0.012648809523834842, 0.009050518347859719
        , 0.008547008547008548, 0.007246376811594203, 0.00735510444248841, 0.006172839506172839, 0.006007067137814506
        , 0.005264854410660672, 0.0048625792811857055, 0.00485725614592042, 0.004638715432651846, 0.004267668146126967
        , 0.0038722168441440064, 0.0038750678136875256, 0.0036185268575112364, 0.003386739611828028,
         0.003478370857577441
        , 0.0032711179526658737, 0.002996948561464995, 0.0029446407538284438, 0.002875359419928197,
         0.0026607789855075284
        , 0.002968270214944232, 0.0027628326305077088, 0.0027066601588506107, 0.0027255490598836634,
         0.002479874861699664
        , 0.0022442786698698538, 0.0033406352683471303, 0.0030187558766766537, 0.002926530968813371,
         0.004198740377888025
        , 0.003919291624330004, 0.0036730945821868677, 0.005909762473010887, 0.00539568345324086, 0.004469657899259596
        , 0.008219580214302174, 0.0064516129032258064, 0.006076005302698056, 0.018237082066969963, 0.011235955056179775
        , 0.008695652173913044, 0.08333333333333333, 0.018518518518518517, 0.008887821282733764, 0.00641185777333491
        , 0.005806954181716289, 0.004736673089272918, 0.004547669319475254, 0.0041327489041977925, 0.003543525594502076
        , 0.0032106693010625796, 0.0028571428571431408, 0.010018214936254155, 0.008529335745675876, 0.006777691711853619
        , 0.005394316973106612, 0.005181766658573555, 0.004837447535039937, 0.003720478809447923, 0.003408606731998637
        , 0.0032884072067074777, 0.002562771686239291, 0.0024269519399651445, 0.002158462753356263,
         0.0029585798816572943
        , 0.002798507462686896, 0.0025869408605675675, 0.008188331627496683, 0.004853344587466669, 0.0039015931505343364
        , 0.003942686796810331, 0.0035182541301257983, 0.0031455533314292917, 0.04761904762274717, 0.008799718409015563
        , 0.007226428079838609, 0.007219402143263654, 0.0058416270012992925, 0.005477018725486285, 0.0053886469328990405
        , 0.005181347150261582, 0.004504504504505597, 0.00485735366453657, 0.004375034612617045, 0.003892968018529284
        , 0.0040672318322158745, 0.0036764705882363974, 0.0032428742106166077, 0.003468653648510714
        , 0.0031625040544931327, 0.00281950932121405, 0.0029932651534059856, 0.0026118698470643975, 0.002356222350452461
        , 0.0032369684116543922, 0.0030363312742131897, 0.002807805699846198, 0.003909131140288855, 0.00358744394619024
        , 0.0032733224222593446, 0.004340425531916954, 0.004118424119980586, 0.00368464961067914, 0.005277044854882977
        , 0.004824063564133099, 0.004366812227074236, 0.0075559430398288285, 0.005780346820809248, 0.005319148936170213
        , 0.011538461538489182, 0.008947855600137047, 0.007463614877476313, 0.25, 0.03006833713004225
        , 0.014745308310961042, 0.024060785141546113, 0.012434998869521173, 0.011963255714609102, 0.007765733700055215
        , 0.006820877817323953, 0.005676442762532631, 0.00581549035158191, 0.004929069487859939, 0.0046370967741950885
        , 0.004667939741146282, 0.004069451980467687, 0.003902704665094526, 0.003910415926061269, 0.003387643168253648
        , 0.0033358605003802975, 0.0032037590773193775, 0.002905770029058586, 0.0028130977832798105
        , 0.0024557540858671473, 0.0023607589402821266, 0.002230814991077157]
    B = [
        -0.0, -0.0001537515375177391, -0.0, -0.00017126220243485635, -0.0, -0.0001858045336340729, -0.0, -0.0,
        -0.0001977456990349564, -0.0, -0.00026267402154616954, -0.0002442002442062076, -0.0, -0.0,
        -0.00019164430816772027, -0.00019040365575381576, -0.00017488632389253036, -0.0001546551190868335,
        -0.0002995357196390525, -0.00028918449971499687, -0.00025575447570659534, -0.0, -0.00012291052114212034,
        -0.00032644178455197434, -0.00010853049707085448, -0.00021360674997558055, -0.00020012007204522835,
        -0.0002666666666702222, -0.00012388503468934445, -0.00023512814484170151, -0.00017624250969644416,
        -0.00015738117721368243, -0.0002836074872416847, -0.0, -0.0002225684398001907, -0.00019892580068030485, -0.0,
        -0.00035599857601836945, -0.00024801587302202417
        , 0.00016455487905487172, -0.0, -0.0, -0.00014710208885182556, -0.0, -0.00023557126030901735,
        -0.00010744600838194313, -0.00010570824524424639, -0.00029738302934474015, -0.00017841213202656926,
        -0.00017070672584645533, -7.446570854177129e-05, -7.750135627433543e-05, -8.041170794532335e-05,
        -6.512960792018451e-05, -0.00012648621300358264, -0.0001282791353994502, -5.448997384510947e-05,
        -6.54364611966067e-05, -0.00012501562695415062, -5.661231884090021e-05, -0.00010235414534341022,
        -4.84707479038751e-05, -0.0001774859120565181, -0.00015800284405181704, -0.00011445576284809605,
        -3.160955873066004e-05, -0.00016429353778841345, -4.9487801257235056e-05, -0.00013103870009666742,
        -0.00013995801259720055, -0.00014515894905026244, -0.00018365472911039887, -0.00011364927832837425,
        -0.00019984012789967868
        , 8.595495960190782e-05, -0.0001467782181145865, -0.0, -0.00011047282368659383, -0.0006079027355807873, -0.0,
        -0.0, -0.0, -0.0, -0.00012010569301129122
        , 8.327088017389684e-05, -7.081651441166219e-05
        , 8.028259473410632e-05, -0.0001624167614106568, -0.00025046963055886333, -0.00017077231780801252,
        -0.00012348728081058486, -7.142857142882653e-05, -0.00013010668748542943, -0.00012923235978455974,
        -9.682416731309862e-05, -0.00015195259079282749, -0.0002428953121224427, -0.00035569467169634843,
        -0.0001617599482376888, -5.325948018775703e-05, -0.00018526237784347767, -0.00012976059170871914,
        -9.580073447260354e-05, -5.264543300882507e-05, -0.00011678604796059166, -8.111615833907007e-05,
        -0.00013098434737091814, -0.0010235414534550547, -7.033832735507026e-05
        , 0.0001083775875160765, -0.00019232618521196584, -0.00020395676116767263, -0.00019063959584587399,
        -0.006349206349878222, -0.00011732957878818879, -0.00011470520761774153, -0.00011280315848971014,
        -7.211885186839838e-05, -0.00022204129968338423
        , 7.381708127315138e-05, -0.00020187066819327013, -0.0001126126126132467, -0.00012297097884974773,
        -0.0001661405549103735, -9.855615236829723e-05, -0.00019601117263780081, -0.0001838235294126095,
        -8.53387950165785e-05, -0.00017129153819874654, -0.00016217969510283077, -8.416445735001542e-05,
        -0.0002494387627847736, -0.00012587324564203498, -9.617234083523838e-05, -0.000223239200804907,
        -0.00020940215684331172, -0.00014039028499293553, -0.0002950287653067933, -0.00022421524663844707,
        -0.0001169043722242045, -0.00017021276595889545, -0.00023311834641567426, -6.95216907680028e-05,
        -9.772305286912658e-05, -9.458948165053528e-05, -0.0, -0.00043591979076483463, -0.0, -0.0,
        -0.00038461538462278104, -0.00030854674483660205, -0.00012439358129275274, -0.0, -0.0004555808656244
        , 0.0002234137622927483, -0.00042211903758641276
        , 0.00022609088854230364, -0.0002136295663365441, -0.00016178611875362865, -0.00014827995255261388
        , 0.00011825922422088766, -0.000396510705794297, -0.00012022120702236381, -0.00010080645161391942,
        -0.0003182686187175713, -9.043226623340959e-05, -9.076057360764895e-05, -0.00035549235691748575,
        -8.065817067333972e-05, -0.000151630022745653, -0.00028477858465246177, -0.00013837000138465733,
        -0.00016878586699768644, -0.00012702176306260911, -0.00017487103261421209, -0.00014872099940566895]
    X = [3.9448, 4.1474, 3.9827, 4.0165, 4.2275, 4.0023, 3.9861, 4.2017, 4.0891, 3.8547, 4.08, 3.9364, 3.6493, 3.9163
        , 3.7548, 3.6854, 3.8763, 3.6778, 3.6562, 3.7994, 3.5745, 3.2884, 3.4486, 3.2326, 3.0147, 3.1514, 2.9161, 3.0197
        , 3.2133, 2.9908, 3.0772, 3.2239, 3.0773, 3.1232, 3.3433, 3.2117, 2.7291, 2.9845, 2.9108, 2.7236, 2.9859, 2.8604
        , 2.7039, 2.9308, 2.7817, 2.6992, 2.8979, 2.6815, 2.5267, 2.67, 2.4335, 2.4621, 2.586, 2.3428, 2.4101, 2.4854
        , 2.2019, 2.5843, 2.6542, 2.3967, 2.1468, 2.2193, 1.9569, 1.8199, 1.8045, 1.5352, 2.0588, 2.1815, 1.904, 2.2819
        , 2.4595, 2.2173, 2.595, 2.7454, 2.5638, 2.6736, 2.8317, 2.6327, 2.5922, 2.854, 2.7898, 2.2647, 2.2428, 1.8361
        , 2.0774, 1.8831, 1.5757, 1.7584, 1.5323, 1.6628, 1.7747, 1.5, 1.9837, 2.2702, 2.1502, 2.0353, 2.3092, 2.1307
        , 1.9408, 2.1395, 1.8752, 1.5016, 1.613, 1.3351, 1.7161, 1.8627, 1.6069, 1.4053, 1.6117, 1.3652, 1.3422, 1.4881
        , 1.2072, 1.6482, 1.8922, 1.6923, 1.7208, 1.9546, 1.7288, 1.741, 1.958, 1.7187, 1.7555, 1.9239, 1.6727, 1.688
        , 1.839, 1.5774, 1.6049, 1.7091, 1.4125, 1.4104, 1.4581, 1.191, 2.2389, 2.2937, 2.0302, 2.4543, 2.5486, 2.3064
        , 2.5463, 2.6801, 2.443, 2.6111, 2.7776, 2.5356, 2.5905, 2.7927, 2.5727, 2.475, 2.7038, 2.5506, 2.2331, 2.493
        , 2.3579, 2.7076, 2.9692, 2.8304, 2.9521, 3.1708, 2.9723, 3.0106, 3.1986, 2.9787, 2.9608, 3.118, 2.8695, 2.8075
        , 2.8916, 2.6856, 2.5519, 2.602, 2.3392, 2.0837, 2.0612, 1.8081]
    Y = [-0.21481, -0.37093, -0.54338, 0.29453, 0.13786, 0.014408, 0.69162, 0.57927, 0.34759, 1.1899, 1.0801, 0.87283
        , 0.87604, 0.74994, 0.51765, 0.31327
        , 0.12641, -0.029424, -0.26297, -0.47355, -0.63642, -0.39051, -0.61401, -0.75143, -0.32259, -0.52345, -0.66005
        , 0.15111, -0.0070697, -0.16286, 0.68231, 0.46353, 0.26158, 1.1663, 1.0108, 0.79073, 1.3385, 1.2957, 1.0678
        , 0.82345, 0.75352, 0.56497, 0.45302, 0.30057
        , 0.10687, -0.068576, -0.22533, -0.3976, -0.37464, -0.62066, -0.71665, -0.75624, -0.99587, -1.062, -1.0925,
         -1.3423, -1.4132, -1.7665, -2.001, -2.0352, -1.4674, -1.7139, -1.7414, -1.5636, -1.8248, -1.8065, -1.0813,
         -1.3297, -1.3507, -0.54441, -0.77107, -0.89852
        , 0.11477, -0.085686, -0.2443, 0.57658, 0.37366, 0.17772, 1.0546, 0.99097, 0.75592, 1.2787, 1.0205, 0.42951
        , 0.29868, 0.083287, -0.085805, -0.29208, -0.45125, -0.73064, -0.98523, -1.0511, 0.57471, 0.53108
        , 0.28373, -0.039897, -0.16609, -0.36664, -0.72169, -0.92532, -1.041, -1.4164, -1.689, -1.6867, -1.1585,
         -1.4043, -1.4637
        , 0.062155, -0.13473, -0.25945, -0.44658, -0.685, -0.72624, 0.58303, 0.46585, 0.25335, 0.26783
        , 0.097811, -0.063996
        , 0.056777, -0.11801, -0.26503, -0.19815, -0.3772, -0.4926, -0.48649, -0.73282, -0.78229, -0.75005, -1.0014,
         -1.0143, -1.0379, -1.3128, -1.2765, -1.3469, -1.6072, -1.5867, -0.89645, -1.1553, -1.1943, -0.52996, -0.75561,
         -0.84442, -0.051185, -0.24842, -0.37595
        , 0.35713, 0.18851, 0.039107, 0.77974, 0.64907, 0.4731, 1.3477, 1.2588, 1.04, 1.3086, 1.208, 1.0012, 0.5653
        , 0.43714
        , 0.27003, -0.041549, -0.23006, -0.35373, -0.5449, -0.76302, -0.82701, -1.1188, -1.3515, -1.3709, -1.6379,
         -1.8916, -1.8947, -2.2073, -2.4832, -2.3963]

    k = np.zeros((9, 9))
    fu = 250.001420127782
    fv = 253.955300723887
    u0 = 239.731339559399
    v0 = 246.917074981568
    H = compute_H(A, B, X, Y, fu, fv, u0, v0)
    print(H)
    H1 = optimizeH(H, A, B, X, Y, fu, fv, u0, v0)
    print(H1)

    H2 = Hoptimizationlsqnonlin(H1, A, B, X, Y, fu, fv, u0, v0)
    print(H2)
