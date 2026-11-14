"""The take-home preserves its finite-window model, inference and evidence contracts."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.mmar_takehome import diagnostics, inference, lesson, model
from sbi4f_tutorial.shared import path_statistics


# Generated from the pinned upstream sensitivity simulator at revision 069d5dc
# with NumPy 2.5.3: three paths, seed=701+d. Fixed coordinates include block
# boundaries and the phase-shifted endpoints. Whole-row projections guard the
# remaining observations. Tolerances permit platform libm rounding; the authoring
# evaluation separately compares every output bit with upstream on one platform.
PROJECTION_DAYS = [0, 1, 31, 32, 63, 64, 127, 128, 191, 192, 254, 255]
SOURCE_FIXTURES = {
    4: {
        'returns': [
            [-0.002663668936237388, -0.012550087752102281, 0.00040167068863646883, -0.003921060077500141, -0.0018498262936324913, 0.0008861859730172354, -0.0012836090118712557, -0.00169643231567582, 0.04032502618729384, -0.010749107659003962, -0.0012797018544022211, -0.00037071312032826, -0.22601133027663262, 0.025897401182570445, -0.11972500999211491],
            [-0.0012896247668211394, 0.0005942017234580194, -0.0005727118173616012, 0.00010405677318000291, 0.00010739703632608323, -6.62026530422894e-05, -0.0023041170619818095, 0.002006444199649868, -0.004969797438581132, -0.0013788878068397188, 0.001463753934152196, 0.0001934406332014247, 0.001996989955213335, 0.0026931086872725176, 0.0056176006220990705],
            [-0.007032466167352031, -0.01759468786723701, 0.0012555918765291716, -0.0023748051880829813, 0.0014137924069867213, 0.01739814105606617, 0.001786984416690545, 0.015204385831133777, -0.0198187696932048, 0.022858026301792674, 0.0053839705771564945, 0.007838110789309207, 0.7110656249573026, 0.06044549319998413, 0.45922042004944286],
        ],
        'trading_time_increments': [
            [0.9757267814651998, 0.9757267814651998, 0.7126759237147053, 0.7126759237147053, 0.7126759237147053, 0.7126759237147053, 0.9757267814651998, 0.9757267814651998, 1.3358705133548952, 1.3358705133548952, 0.9757267814651998, 0.9757267814651998, 256.0, 268.5787203091555, 137.81573653380343],
            [0.7296910405479795, 0.7296910405479795, 0.7296910405479795, 0.7296910405479795, 0.23048405216806728, 0.23048405216806728, 2.3101338667359737, 2.3101338667359737, 0.7296910405479795, 0.7296910405479795, 0.7296910405479795, 0.7296910405479795, 256.0, 413.1033222308031, 138.1091655707907],
            [0.46390933235092197, 0.46390933235092197, 0.07172641289675989, 0.07172641289675989, 0.07172641289675989, 0.07172641289675989, 0.46390933235092197, 0.46390933235092197, 3.000454922401396, 3.000454922401396, 0.46390933235092197, 0.46390933235092197, 256.0, 604.0510820450795, 169.9987494755834],
        ],
    },
    8: {
        'returns': [
            [-0.009548122952161623, 0.0022032820786208936, -0.0036112262518745867, -0.009378888932781198, 0.004116362478771486, 0.0052135044720959, -0.002988512055289533, 0.005264866795468365, 0.0016196598264589875, 0.005625649820392717, -0.0029195461223371203, -0.009008401258907279, 0.32433858322257636, 0.014056406322190367, 0.1622969643453281],
            [-0.0018706625135379428, -0.0011741025018505993, -0.001922298032163158, -0.001630683760428277, 0.002076178096928517, 0.0004969300385497944, 0.010835162851262672, 0.0011093523046815394, 0.0007010435206965885, 8.028915895319473e-05, 0.0027453301404406407, 0.0010668985792076065, 0.3572782293066913, 0.03838842411682508, 0.1782516479274759],
            [0.0032473357648246494, 0.0003809879995773243, -0.004090461460492413, -0.0002768533799504634, -0.0014678644109976256, -0.004067445964924534, -0.005829649510967345, 0.0006574286355358833, -0.0006720342529855966, -0.0031487800469271712, -0.0030315319408900352, -0.0032992972329850155, -0.25727734103019856, 0.13262209225105087, -0.21847079027574523],
        ],
        'trading_time_increments': [
            [1.4237735826007483, 1.4237735826007483, 1.1074117741429763, 1.1074117741429763, 0.8613454080741909, 0.8613454080741909, 0.6699548707477502, 0.6699548707477502, 1.1074117741429763, 1.1074117741429763, 1.4237735826007483, 1.4237735826007483, 256.00000000000006, 268.1856317412279, 125.80702151994204],
            [0.13095019433739521, 0.13095019433739521, 0.13095019433739521, 0.13095019433739521, 0.8194488637324461, 0.8194488637324461, 0.819448863732446, 0.819448863732446, 0.13095019433739521, 0.13095019433739521, 0.13095019433739521, 0.13095019433739521, 256.0, 907.5676718068974, 97.48561326972623],
            [1.1842712245994762, 1.1842712245994762, 0.5642497369534424, 0.5642497369534424, 0.5642497369534424, 0.5642497369534424, 1.1842712245994762, 1.1842712245994762, 2.485598559579066, 2.485598559579066, 1.1842712245994762, 1.1842712245994762, 256.00000000000006, 365.219285300638, 150.28143846232874],
        ],
    },
    16: {
        'returns': [
            [0.0014003541592502921, -0.004942293580853149, -0.0012438829584247104, 0.004866214372414535, 0.0008436851462783604, 0.004500050716087159, 0.0009951283909473024, -0.0009761588322257369, 7.222551816598612e-05, -0.0029280332773893285, 0.001806038866355417, 0.0007608269756918469, 0.194032979719009, 0.015403591542085067, 0.07822428198349421],
            [0.017519890247152795, -0.0013187911278856965, -0.002237766831682935, -0.0007219464553089947, -0.004325947396114951, 0.00116963782873298, -0.0028316609472363793, -0.001486371242119637, 0.011638605044008484, 0.013032060930427634, -0.0003189389923068893, -0.016050958127404532, 0.08592143361326843, 0.05132668552916278, 0.06191980761533386],
            [0.004784640179837853, 0.00691893495052314, -0.005697109485389767, 0.0010195469643845536, 0.0044545970366490495, 0.008019681016173733, 0.0033824544722955664, 0.0032819289388512827, 0.007634171675890696, -0.003396357434245852, 0.001432740300669978, 0.0059235620610502415, 1.011236425557815, 0.016553513659802038, 0.5509327615385851],
        ],
        'trading_time_increments': [
            [0.028650861253613245, 0.028650861253613245, 1.6493485880172976, 1.6493485880172976, 0.11062737446762486, 0.11062737446762486, 0.11062737446762486, 0.11062737446762486, 1.6493485880172973, 1.6493485880172973, 0.028650861253613245, 0.028650861253613245, 256.00000000000006, 841.34064397024, 143.7597811679694],
            [1.6854725911590616, 1.6854725911590616, 0.2096184426706504, 0.2096184426706504, 1.6854725911590618, 1.6854725911590618, 0.07392364758335537, 0.07392364758335537, 1.6854725911590618, 1.6854725911590618, 1.6854725911590616, 1.6854725911590616, 256.00000000000006, 584.1025543399314, 108.66191505100713],
            [0.3639205622005, 0.3639205622005, 1.602549872745911, 1.602549872745911, 0.3639205622005, 0.3639205622005, 0.0826421553829086, 0.0826421553829086, 0.3639205622005, 0.3639205622005, 0.3639205622005, 0.3639205622005, 256.0, 974.3261914141638, 184.48033535220756],
        ],
    },
    32: {
        'returns': [
            [-0.00139909651745861, 0.0011438176942231092, -0.00387180781200873, 0.00197641112137033, 0.003938600830898718, -0.0032915800920501376, 0.000891733739952376, -0.006261677772787536, -0.0011627852612899649, -0.001973344146027653, 0.0029740762732328556, -0.0015122987266335272, 0.0018518861072775606, 0.0040614532068757895, 0.021439872653547448],
            [-0.02044317862085782, 0.009993112856973473, -0.008598327128621575, -0.016926116078370756, 0.01052891315326146, 0.02825244534525822, 0.007544311152727584, 0.00035551067870438465, 0.0187421619015803, 0.045500295025863904, 0.0029779548774426135, 0.005327394917529283, 0.6186366081927213, 0.13447375716586463, 0.2886084489742962],
            [-0.024090878296267305, -0.014879213695843272, -0.003983166742901235, 0.0065736927854311705, -0.0035138504781162955, -0.006468027891990975, 0.0011949282371392753, -0.02380787973479828, 0.013258145444002337, -0.020338270310724063, 0.009654879835801097, 0.024586781735802306, -1.0619886604830375, 0.06773553536955598, -0.5207318403305985],
        ],
        'trading_time_increments': [
            [0.24840972973922482, 0.24840972973922482, 0.24840972973922482, 0.24840972973922482, 0.7989976900214604, 0.7989976900214604, 0.7989976900214603, 0.7989976900214603, 0.24840972973922482, 0.24840972973922482, 0.7989976900214603, 0.24840972973922482, 256.0, 867.058612582782, 158.35669812381153],
            [1.1058018105171976, 1.1058018105171976, 0.7274232489278278, 0.7274232489278278, 0.7274232489278278, 0.7274232489278278, 1.680998848944452, 1.680998848944452, 0.478516654655529, 0.478516654655529, 1.1058018105171976, 1.1058018105171976, 255.99999999999994, 315.3777091171421, 128.7185306113719],
            [1.06787079255121, 1.06787079255121, 0.2914534213703685, 0.2914534213703685, 0.2914534213703685, 0.2914534213703685, 0.5578840346080332, 0.5578840346080332, 2.044059264727217, 2.044059264727217, 1.06787079255121, 1.06787079255121, 256.00000000000006, 409.30611097168094, 143.77164792573632],
        ],
    },
    64: {
        'returns': [
            [0.0001714456508447146, -0.028283018236696957, 0.008222572826275032, -0.003877183130184146, -0.00383402235107134, -0.02539425587673353, -0.0021192057446995675, -0.011803019519579058, -0.001830535662502525, -0.007656727820731682, -0.004830085413779576, -0.026972471827550543, -0.7007816538715095, 0.0446211485082696, -0.24832109986618156],
            [0.0037363180788961233, -0.0006052890553464923, 0.01675166903705065, 0.001223616924040181, 0.004673473579104136, -0.0018308202780915478, 0.001237373977987114, -0.0018741838679130584, -0.0003696375107835395, -0.003118886453300992, -0.002422123842298604, 0.004125425597608839, 0.07902639590684303, 0.025299913691142113, 0.014930263635912017],
            [0.0012200270286488795, -0.0008772618590330408, 0.0004322237056081767, 0.0002576656185984573, -0.0010706450210607246, -0.0026594579352690563, -0.0007541437057907418, -0.006031935129591424, 0.0004143533015572885, 0.0006131581362290239, 0.0007738875577595059, 0.0003467551684476225, 0.08155069710956049, 0.014455787976845408, 0.031423796455538625],
        ],
        'trading_time_increments': [
            [0.3373051452422352, 1.16606281906164, 0.33730514524223526, 0.33730514524223526, 1.16606281906164, 1.16606281906164, 1.16606281906164, 1.16606281906164, 0.09757172525099705, 0.09757172525099705, 0.3373051452422352, 0.3373051452422352, 256.0, 1258.0475793203177, 155.07909696494204],
            [0.2691221829365381, 0.2691221829365381, 4.181598181724941, 0.269122182936538, 1.0608302554269813, 0.06827364602207342, 0.269122182936538, 0.269122182936538, 0.01732035126307876, 1.0608302554269813, 0.06827364602207342, 0.06827364602207342, 255.99999999999997, 1580.0304339891304, 109.57337020504335],
            [0.029760730855694396, 0.15595703350886983, 0.029760730855694396, 0.029760730855694396, 0.1559570335088698, 0.1559570335088698, 0.8172714715516765, 0.8172714715516765, 0.005679135343483613, 0.005679135343483613, 0.029760730855694396, 0.029760730855694396, 256.0, 2497.1214079117144, 121.39360441976984],
        ],
    },
    128: {
        'returns': [
            [-0.008016437360411289, 0.0001439244605632656, 0.0027978462578294215, -0.00039449591557849293, -0.009369077535245874, -0.010347703112855463, 0.006048127666242768, 0.00503779419253006, -0.014664908552603861, -0.008814202461564544, -0.022213865200623042, -0.038572614699820595, -0.21842495845232276, 0.0617194759432141, -0.08971324906522862],
            [-0.0006385589107603246, -0.0006191659406919647, -0.0009546583693201829, 3.134449148864757e-05, -0.0009277420123121607, -0.0002655764827163426, -0.006372819789981932, -0.006148687384677264, -0.002019952666367584, 0.0003613327983918803, 0.000358426925406188, -0.00016104906454342355, -0.07867446981951512, 0.003033549194793283, -0.04534952789910176],
            [-0.00012724941065072552, 0.0008468419142856824, -0.0014656252328306384, 0.0011492418070817257, 0.033624961120810405, -0.016923922111860026, 0.001557420308257508, 0.006800831647890391, 0.015385640655246327, 0.009314825594612091, -0.0048571677875492245, 0.0005379469037069513, 0.8091745515484692, 0.06729276084842001, 0.418221621543323],
        ],
        'trading_time_increments': [
            [0.7676019862450935, 1.0723099241409688, 0.7676019862450935, 0.7676019862450935, 1.4979749844525105, 1.4979749844525105, 1.0723099241409688, 1.0723099241409688, 0.5494799553957623, 0.5494799553957623, 0.3933395519970996, 0.7676019862450935, 256.0, 309.3825991564833, 117.53844320003384],
            [0.04129207904339984, 0.04129207904339985, 0.04129207904339984, 0.04129207904339984, 0.007378787885606503, 0.007378787885606503, 0.23107261221756, 0.23107261221756, 0.23107261221755995, 0.23107261221755995, 0.23107261221755995, 0.04129207904339984, 255.99999999999997, 4086.740927180507, 115.69118346177498],
            [0.031042801443779886, 0.008136699689819222, 0.11843321717817425, 0.11843321717817425, 1.7238471943019715, 1.7238471943019715, 0.11843321717817425, 0.11843321717817425, 0.45184153100921537, 0.45184153100921537, 0.11843321717817422, 0.031042801443779886, 256.0, 2004.8264508785858, 115.76126910813069],
        ],
    },
}
MIXED_SOURCE_RETURNS = [
    [-0.010297370689982212, -0.022154606678379932, 0.017132209745429273, -0.00033911596817589283, 0.0005847229570409452, 0.0005036934783822871, -0.0017045493782925736, -0.0006564269286796843, -0.003102861992298465, -0.003467161421810047, 0.0032311374840551287, -0.0014645059436027602, -0.7654736163118643, 0.06357396883564857, -0.4576143004744957],
    [-0.006174559300169641, 0.009250561520812517, 0.0023676412817510346, -0.009833972114889141, 0.0022333699600122652, 0.014770472077684355, 0.024279658645863445, 0.01160541220709597, 0.016000232254552597, -0.00843226972682554, 0.0037370243197888814, 0.005931085793684708, 0.5396994446964148, 0.1566613248159196, 0.2649931816471755],
    [-0.0032152501834920807, -0.0045026425159751015, 0.006517780741064347, 0.018296868213890424, 0.00033164237555072186, 0.0014094683157284459, 0.009638482334134559, 0.07241282377105121, -0.0022290608145511598, -0.007016538515101088, 0.003170032841010543, -0.00110842735661228, -0.9990844712349195, 0.13727984264518034, -0.5645251683505721],
    [-0.0008876909959556115, -0.0008205021559381514, -0.0035409722931768394, -0.0038668153975193894, 0.0009142485767838766, -0.003820885183488499, -0.0014700613921357238, -0.0005622540638743445, 0.009930976578476648, 0.0010611850129557607, -0.003974080142442194, -0.0008572806612978252, -0.3888288789489378, 0.04829102103082131, -0.1994147812811609],
    [-0.00327523192556557, -0.002841956408435142, -0.00342628065752473, -0.003980683341007761, -0.003435089989719813, -0.0026597942318950817, -0.0031577139153702092, -0.0016345559532002464, -0.003201048911169598, -0.00566774714331458, 0.00017133687994163975, -0.0035349331277289703, -0.7627273558092031, 0.019006994687209375, -0.3734366723294573],
    [0.011108607553881197, -0.01348970328947734, 0.003825759325768694, 0.005057244878338931, -0.01267680181080072, 0.002517268432983239, -3.724327804154742e-05, -0.0021466996701385377, -0.002080861694288125, -0.004649328935506499, -0.003246945037411619, -0.01718134387262849, 0.4145416242398587, 0.014907506513118592, 0.1870631184239132],
    [-0.004410853378706767, -0.001760861817619216, 0.0018416978347275308, 0.005826273575579265, -0.007137625583681732, -0.0007387232889621554, 0.005271022347694817, -0.005476778396912027, 0.0018181740745869374, 0.0024528795603616154, -0.013057623404335488, -0.004329757407558836, 0.07768679171256865, 0.014343951217531155, 0.06301610380534768],
    [0.0016118715014511364, 0.000539242541154455, 0.0012968527746840639, 0.0021598040945311646, 0.0012921014442406594, 0.001601916509089537, 0.0011439081561512897, 0.0008123346084542378, -0.01557538124200462, 0.0030770385869003064, 0.0010478205511984672, 0.0008953905692161709, 0.36330678540336614, 0.00248782993873239, 0.19262364456762845],
]


def source_projection(values):
    return np.column_stack((values[:, PROJECTION_DAYS], values.sum(axis=1),
                            np.sum(values*values, axis=1), values@np.linspace(1/256, 1, 256)))

def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


class MMARModelTests(unittest.TestCase):
    def test_all_supported_resolutions_match_pinned_source_fixtures(self):
        for d, expected in SOURCE_FIXTURES.items():
            with self.subTest(d=d):
                values = model.simulate(3, d, seed=701+d)
                self.assertEqual(values["returns"].shape, (3, 256))
                for key in ("returns", "trading_time_increments"):
                    np.testing.assert_allclose(source_projection(values[key]), expected[key],
                                               rtol=2e-13, atol=1e-15)
                np.testing.assert_allclose(values["trading_time_increments"].mean(axis=1), 1., atol=3e-16)
                self.assertTrue((values["returns"] > -1).all())

    def test_mixed_design_draw_order_and_actual_prior_are_preserved(self):
        values = model.simulate(8, seed=701)
        np.testing.assert_array_equal(values["d"][:, 0], [8, 32, 32, 128, 64, 4, 16, 64])
        np.testing.assert_allclose(source_projection(values["returns"]), MIXED_SOURCE_RETURNS,
                                   rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(model.parameter_array(values)[:2], [
            [-.0014700964745136263, .023456272813852496, .8729029292091803, 2.2520798719010076],
            [.00014545674052041988, .024079913681858134, .65587247761778, 2.630590904737756]], rtol=2e-13, atol=1e-15)
        prior = model.prior(10000, np.random.default_rng(37))
        # The tail prior is uniform in log(nu-2), not in nu itself.
        mapped = (np.log(prior[:, 3]-2)-np.log(.05)) / np.log(10/.05)
        self.assertLess(abs(mapped.mean()-.5), .01)
        self.assertGreater(np.min(prior[:, 3]), 2.05)
        self.assertLess(np.max(prior[:, 3]), 12.)

    def test_invalid_last_day_redraws_whole_path_and_cascade_but_keeps_parameters(self):
        class ForcedTailRNG:
            def __init__(self):
                self.base = np.random.default_rng(4)
                self.shapes, self.phases = [], []

            def random(self, shape):
                return self.base.random(shape)

            def integers(self, *args, **kwargs):
                self.phases.append(kwargs["size"])
                return self.base.integers(*args, **kwargs)

            def standard_t(self, degrees, *, size):
                self.shapes.append(size)
                values = np.full(size, .2 if len(self.shapes) == 1 else .1)
                if len(self.shapes) == 1:
                    values[0, -1] = -1e6  # A shorter prefix alone would be admissible.
                return values

        rng = ForcedTailRNG()
        parameters = np.tile([0., .015, .7, 4.], (2, 1))
        values = model.simulate_parameters(parameters, 8, rng=rng)
        self.assertEqual(rng.shapes, [(2, 256), (1, 256)])
        self.assertEqual(rng.phases, [(2, 1), (1, 1)])
        recovered = values["returns"] / (.015*np.sqrt(values["trading_time_increments"])*np.sqrt(.5))
        np.testing.assert_allclose(recovered[0], .1)
        np.testing.assert_allclose(recovered[1], .2)
        np.testing.assert_array_equal(model.parameter_array(values), parameters)

    def test_known_condition_shapes_exclude_parameters_and_latent_weights(self):
        pairs = model.simulate(2, [4, 128], seed=8)
        conditions = model.conditions(pairs["returns"], pairs["d"][:, 0])
        self.assertEqual(set(conditions), {"returns", "log_d"})
        self.assertEqual(conditions["returns"].shape, (2, 256, 1))
        self.assertEqual(conditions["log_d"].shape, (2, 1))
        np.testing.assert_allclose(conditions["log_d"][:, 0], np.log([4, 128]), rtol=1e-7)
        for bad in (np.zeros(255), np.zeros((2, 128)), np.full(256, -1.),
                    np.full(256, np.nan), np.zeros(256, dtype=complex),
                    np.ma.array(np.zeros(256), mask=True)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                model.conditions(bad)
        for d in (None, 3, 7, 8.5):
            with self.subTest(d=d), self.assertRaises(ValueError):
                model.conditions(np.zeros(256), d)

    def test_named_sampling_order_and_support_are_checked_without_neural_backend(self):
        estimator = mock.Mock()
        output = {name: np.full((2, 3, 1), value) for name, value in
                  reversed(list(zip(model.PARAMETERS, [0., .015, .7, 4.]))) }
        output["_summaries"] = np.zeros((2, 16))
        estimator.sample.return_value = output
        rows = inference.sample(estimator, np.zeros((2, 256)), [4, 8], count=3)
        np.testing.assert_array_equal(rows[0, 0], [0., .015, .7, 4.])
        self.assertEqual(set(estimator.sample.call_args.kwargs["conditions"]), {"returns", "log_d"})
        estimator.sample.return_value = dict(output, q=np.full((2, 3, 1), .99))
        with self.assertRaisesRegex(ValueError, "support"):
            inference.sample(estimator, np.zeros((2, 256)), [4, 8], count=3)
        for count in (0, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                inference.sample(estimator, np.zeros(256), count=count)

    def test_simple_return_drawdown_uses_log1p_and_initial_wealth(self):
        simple = np.array([[-.1, .05, -.2], [.1, .2, .05]])
        wealth = np.column_stack((np.ones(2), np.cumprod(1+simple, axis=1)))
        expected = np.max(1-wealth/np.maximum.accumulate(wealth, axis=1), axis=1)
        np.testing.assert_allclose(path_statistics.maximum_drawdown(np.log1p(simple)), expected)
        np.testing.assert_allclose(lesson.summary(simple)[:, 2], 100*expected)

    def test_saved_and_fresh_dependence_checks_all_use_absolute_log_returns(self):
        report, data = lesson.load_saved()
        _, _, observed = lesson.load_market()
        expected = path_statistics.sample_acf(np.log1p(observed), absolute=True)["values"][0]
        np.testing.assert_allclose(data["observed_abs_acf"][0], expected)
        prior = model.simulate(200, 8, seed=report["seeds"]["prior"])
        expected = path_statistics.sample_acf(np.log1p(prior["returns"]), absolute=True)["values"]
        np.testing.assert_allclose(data["prior_abs_acf"], expected, rtol=2e-13, atol=1e-15)
        simple = np.stack([np.linspace(-.4, .4, 256), np.sin(np.arange(256)*.2)*.2])
        parameters = np.tile([0., .015, .7, 4.], (2, 1))
        with mock.patch.object(model, "simulate_parameters", return_value={"returns": simple}):
            result = lesson.scenario_result(parameters)
        expected = path_statistics.sample_acf(np.log1p(simple), absolute=True)["values"]
        np.testing.assert_allclose(result["abs_acf"], expected)
        self.assertGreater(np.max(np.abs(expected-path_statistics.sample_acf(simple, absolute=True)["values"])), .01)


class MMARDiagnosticTests(unittest.TestCase):
    def test_discrete_rank_null_includes_both_endpoints(self):
        statistic, deviation = diagnostics.rank_statistic([0, 1, 2, 3], 3)
        self.assertEqual(statistic, 0.)
        np.testing.assert_array_equal(deviation, np.zeros(4))
        statistic, deviation = diagnostics.rank_statistic([0, 0, 0, 0], 3)
        self.assertEqual(statistic, .75)
        np.testing.assert_allclose(deviation, [.75, .5, .25, 0.])
        for ranks in ([-1, 0], [0, 4], [.5, 1]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                diagnostics.rank_statistic(ranks, 3)

    def test_holm_adjustment_preserves_display_order_and_monotonicity(self):
        rows = [{"parameter": name, "rank_p_value": value} for name, value in
                [("a", .2), ("b", .001), ("c", .02)]]
        adjusted = diagnostics.holm(rows)
        np.testing.assert_allclose([r["rank_holm_p"] for r in adjusted], [.2, .003, .04])
        self.assertEqual([r["rank_holm_reject"] for r in adjusted], [False, True, True])
        self.assertNotIn("rank_holm_p", rows[0])

    def test_canned_bad_estimator_is_detected_and_recovery_baseline_uses_actual_prior(self):
        truth = np.array([[.0001, .012, .6, 3.], [-.0002, .014, .7, 4.],
                          [.0003, .016, .8, 5.], [-.0004, .018, .75, 6.]])
        posterior = truth[:, None, :] + np.arange(1, 4)[None, :, None]*np.array([.001, .001, .01, .1])
        result = diagnostics.evaluate(posterior, truth, null=np.linspace(0., .2, 10000))
        np.testing.assert_array_equal(result["ranks"], np.zeros((4, 4)))
        records = [dict(row, d=d) for d in model.CAPSTONE_D for row in result["records"]]
        self.assertTrue(all(row["rank_holm_reject"] for row in diagnostics.holm(records)))
        self.assertTrue(all(row["coverage90"] == 0 for row in result["records"]))
        prior_nu_mean = 2+(10-.05)/np.log(10/.05)
        expected = np.mean((np.median(posterior, axis=1)[:, 3]-truth[:, 3])**2) / np.mean((prior_nu_mean-truth[:, 3])**2)
        self.assertAlmostEqual(result["records"][3]["mse_to_prior_mean_ratio"], expected)
        rule = inference.freeze_rule()
        self.assertEqual(rule["resolutions"], list(model.CAPSTONE_D))
        self.assertIn("discrete uniform", rule["rank_null"])
        self.assertIn("Holm", rule["decision"])
        self.assertIn("no second pass/fail", rule["coverage"])


class MMARBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "takehomes" / "mmar-v2"
        self.bundle.mkdir(parents=True)
        parameters = np.array([[0., .015, .7, 4.], [.001, .02, .8, 3.]])
        arrays = {}
        for d in model.CAPSTONE_D:
            arrays[f"posterior_{d}"] = parameters.copy()
            arrays[f"mdd_{d}"] = np.linspace(0., 1., 512)
        np.savez_compressed(self.bundle / "saved.npz", **arrays)
        self.market = self.root / "market-examples" / "voo-covid-256"
        self.market.mkdir(parents=True)
        self.returns = np.linspace(-.04, .04, 256)
        self.dates = (np.datetime64("2020-01-01")+np.arange(256)).astype(str)
        np.savez_compressed(self.market / "returns.npz", simple_returns=self.returns, dates=self.dates)
        (self.market / "market.json").write_text(json.dumps({"ticker": "fixture"}))
        self.report = {"format": "mmar_takehome_bundle_v1",
                       "model_design": json.loads(json.dumps(model.DESIGN)),
                       "frozen_rule": inference.freeze_rule(),
                       "saved_sha256": hashlib.sha256((self.bundle / "saved.npz").read_bytes()).hexdigest(),
                       "market": {"simple_returns_sha256": array_hash(self.returns),
                                  "start_date": str(self.dates[0]), "end_date": str(self.dates[-1])}}
        self.write_report()
        self.patch = mock.patch.object(lesson, "ASSETS", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def write_report(self):
        (self.bundle / "report.json").write_text(json.dumps(self.report))

    def test_saved_arrays_are_bound_to_their_report_and_current_design(self):
        report, data = lesson.load_saved()
        self.assertEqual(report["model_design"]["window"], 256)
        self.assertEqual(data["posterior_8"].shape, (2, 4))
        original = copy.deepcopy(self.report)
        self.report["model_design"]["prior"]["q"]["bounds"] = [.5, .99]
        self.write_report()
        with self.assertRaises(ValueError):
            lesson.load_saved()
        self.report = copy.deepcopy(original)
        self.report["frozen_rule"]["resolutions"] = [4, 8, 32]
        self.write_report()
        with self.assertRaises(ValueError):
            lesson.load_saved()
        self.report = copy.deepcopy(original)
        self.write_report()
        (self.bundle / "saved.npz").write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "bound report"):
            lesson.load_saved()

    def test_observed_returns_and_dates_must_match_cached_posterior_inputs(self):
        _, dates, returns = lesson.load_market()
        np.testing.assert_array_equal(returns, self.returns)
        np.testing.assert_array_equal(dates, self.dates)
        changed = self.returns.copy()
        changed[7] += .001
        np.savez_compressed(self.market / "returns.npz", simple_returns=changed, dates=self.dates)
        with self.assertRaisesRegex(ValueError, "bound inputs"):
            lesson.load_market()
        changed_dates = self.dates.copy()
        changed_dates[0] = "2019-12-31"
        np.savez_compressed(self.market / "returns.npz", simple_returns=self.returns, dates=changed_dates)
        with self.assertRaisesRegex(ValueError, "bound inputs"):
            lesson.load_market()

    def test_restore_rejects_wrong_path_or_hash_before_building_a_backend(self):
        outside = self.root / "outside.keras"
        outside.write_bytes(b"fixture")
        cached = self.root / "local-checkpoints" / "mmar" / "model.keras"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"fixture")
        with mock.patch("sbi4f_tutorial.shared.paths.ASSETS", self.root), mock.patch.object(inference, "build_workflow") as build:
            with self.assertRaisesRegex(ValueError, "inside"):
                inference.load_checkpoint(outside, hashlib.sha256(b"fixture").hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                inference.load_checkpoint(cached, "0"*64)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
