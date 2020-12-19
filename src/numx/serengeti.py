import numpy as np
import cv2

from scipy.interpolate import interp1d
from affordable.game import AbstractGame
from numx.tribe import Tribe
from numx.shaman import Shaman
from numx.chief import Chief


class Serengeti(AbstractGame):
    def __init__(self, ctx, alpha=0.1, mode='hidden', device='cpu'):
        super(Serengeti, self).__init__(ctx, 'serengeti')
        self.device = device
        self.mode = mode

        size = self.ctx['size'] if 'size' in self.ctx else 64
        dt = self.ctx['dt'] if 'dt' in self.ctx else 0.0001
        if 'size' not in self.ctx:
            self.ctx['size'] = size
        if 'dt' not in self.ctx:
            self.ctx['dt'] = dt

        self.size = size
        self.dt = dt

        fn = interp1d(np.linspace(0, 1, 25), np.array([
            0.0012761865103869542,
            0.48332041509851986,
            1.0342424980051284,
            1.555750908015132,
            2.101637833028677,
            2.500696997846033,
            2.9682970321872033,
            3.600388676937193,
            4.388440434032988,
            4.693335991974876,
            5.011802148508649,
            5.678951808009407,
            6.187839306647397,
            6.6117418130147065,
            6.675732372269458,
            7.706763660679444,
            7.53567648825326,
            8.028857636959513,
            8.889334475587527,
            10.293711508641234,
            9.525388833221232,
            11.351871412188059,
            10.923061388620157,
            8.472273093952703,
            378.2353736422377,
        ]))

        self.alpha = alpha
        self.beta = 1.0 / size / size * np.exp(3 * 3) * 4 / np.pi / (1.0001 - alpha) * (0.0001 + alpha) / fn(alpha) / 4
        self.steps = 0
        self.accum = 0

        self.gen_peak()
        self.gen_index(size)

        self.berries_hl = np.zeros((size, size))
        self.berries_hr = np.zeros((size, size))
        self.berries_rl = np.zeros((size, size))
        self.berries_rr = np.zeros((size, size))
        self.canvas_hl = np.zeros((size, size))
        self.canvas_hr = np.zeros((size, size))
        self.canvas_rl = np.zeros((size, size))
        self.canvas_rr = np.zeros((size, size))
        self.navi_map = np.zeros((4 * size + 3, 4 * size + 3))
        self.score_img = np.zeros((self.size // 2, 6 * size + 8), dtype=np.uint8)

        tribex = np.random.random() * (self.xmax - self.xmin) + self.xmin
        tribey = np.random.random() * (self.ymax - self.ymin) + self.ymin
        self.tribe = Tribe(size, size, tribex, tribey, 0.0)

    def set_boundaries(self):
        size = self.ctx['size'] if 'size' in self.ctx else 64
        xmin = self.ctx['xmin'] if 'xmin' in self.ctx else -3.5
        xmax = self.ctx['xmax'] if 'xmax' in self.ctx else 3.5
        ymin = self.ctx['ymin'] if 'ymin' in self.ctx else -3.5
        ymax = self.ctx['ymax'] if 'ymax' in self.ctx else 3.5
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.size = size

    def gen_peak(self):
        self.peakx = np.random.random() * (self.xmax - self.xmin) + self.xmin
        self.peaky = np.random.random() * (self.ymax - self.ymin) + self.ymin
        if np.sqrt(self.peakx * self.peakx + self.peaky * self.peaky) > 3.5:
            self.gen_peak()

    def gen_index(self, size):
        IX, IY = np.meshgrid(
            np.linspace(self.xmin, self.xmax, num=4 * size + 3),
            np.linspace(self.ymax, self.ymin, num=4 * size + 3),
        )
        self.IX, self.IY = IX, IY
        XS = (IX < (self.peakx + 4 / self.size)) * (IX > (self.peakx - 4 / self.size))
        YS = (IY < (self.peaky + 4 / self.size)) * (IY > (self.peaky - 4 / self.size))
        self.BLK = XS * YS

        R = np.sqrt(IX * IX + IY * IY)
        self.CCL = (R > 3.45) * (R < 3.5)
        self.XAX = (IX > -0.05) * (IX < 0.05)
        self.YAX = (IY > -0.05) * (IY < 0.05)

    def prosperity(self, xx, yy):
        dx = xx - self.peakx
        dy = yy - self.peaky
        p = np.exp(- dx * dx - dy * dy)
        return p

    def probability(self, xx, yy):
        noise = self.alpha * self.beta * (2 * np.random.rand(*xx.shape) - 1) / 2.0
        signal = (1 - self.alpha) * self.beta * self.prosperity(xx, yy)
        return signal + noise

    def collected_berries(self, xx, yy):
        sample = np.random.rand(*xx.shape)
        berries = sample < self.probability(xx, yy)
        return berries

    def score(self):
        shl = np.sum(self.berries_hl)
        shr = np.sum(self.berries_hr)
        srl = np.sum(self.berries_rl)
        srr = np.sum(self.berries_rr)
        st = shl + shr + srl + srr
        self.accum += st

        if self.mode == 'revealed':
            self.score_img = np.zeros((self.size // 2, 6 * self.size + 8), dtype=np.uint8)
            cv2.putText(self.score_img, '%03d' % shl, (4 + 0 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%03d' % shr, (4 + 1 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%03d' % srl, (4 + 2 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%03d' % srr, (4 + 3 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%03d' % st, (4 + 4 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.score_img, '%05d' % self.accum, (4 + 5 * self.size, self.size // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return st

    def apply_effect(self):
        self.steps += 1
        self.apply_shaman_effect()
        self.apply_chief_effect()
        self.apply_tribe_effect()
        self.score()

        if self.mode == 'revealed':
            self.navi_map[self.BLK] = 1.0
            self.navi_map[self.CCL] = 1.0
            self.navi_map[self.XAX] = 1.0
            self.navi_map[self.YAX] = 1.0

    def apply_tribe_effect(self):
        np.copyto(self.berries_hl, self.collected_berries(*self.tribe.head_left()))
        np.copyto(self.berries_hr, self.collected_berries(*self.tribe.head_right()))
        np.copyto(self.berries_rl, self.collected_berries(*self.tribe.rear_left()))
        np.copyto(self.berries_rr, self.collected_berries(*self.tribe.rear_right()))

    def apply_shaman_effect(self):
        np.copyto(self.canvas_hl, self.shaman_hl.canvas)
        np.copyto(self.canvas_hr, self.shaman_hr.canvas)
        np.copyto(self.canvas_rl, self.shaman_rl.canvas)
        np.copyto(self.canvas_rr, self.shaman_rr.canvas)

    def apply_chief_effect(self):
        sourcex = self.tribe.x
        sourcey = self.tribe.y
        targetx = self.chief.x
        targety = self.chief.y

        self.tribe.x = targetx
        self.tribe.y = targety
        self.tribe.direction = self.chief.direction

        self.tribe.draw_map(sourcex, sourcey, targetx, targety)
        np.copyto(self.navi_map, self.tribe.map)

        print('-----------------------------')
        print(targetx, targety)
        print(self.peakx, self.peaky)
        print((self.peakx - targetx) ** 2)
        print((self.peaky - targety) ** 2)
        print((self.peakx - targetx) ** 2 + (self.peaky - targety) ** 2)
        print(np.sqrt(((self.peakx - targetx) ** 2 + (self.peaky - targetx) ** 2)), self.prosperity(targetx, targety))

    def reset(self):
        super(Serengeti, self).reset()
        self.tribe.clear_map()
        self.navi_map = np.zeros((4 * self.size + 3, 4 * self.size + 3))
        self.score_img = np.zeros((self.size // 2, 6 * self.size + 8), dtype=np.uint8)
        self.steps = 0
        self.gen_peak()
        self.gen_index(self.size)

    def all_affordables(self):
        self.set_boundaries()
        x = 2 * (np.random.random() * (self.xmax - self.xmin) + self.xmin)
        y = 2 * (np.random.random() * (self.ymax - self.ymin) + self.ymin)

        self.chief = Chief(self.ctx, x, y)
        self.shaman_hl = Shaman(self.ctx, 'shaman_hl', self.size, self.size)
        self.shaman_hr = Shaman(self.ctx, 'shaman_hr', self.size, self.size)
        self.shaman_rl = Shaman(self.ctx, 'shaman_rl', self.size, self.size)
        self.shaman_rr = Shaman(self.ctx, 'shaman_rr', self.size, self.size)

        return self.chief, self.shaman_hl, self.shaman_hr, self.shaman_rl, self.shaman_rr

    def state_space(self):
        self.apply_effect()
        x = np.ones((1, self.size * 2 + 3)) * 0.25
        y = np.ones((self.size, 1)) * 0.25
        x2 = np.ones((1, self.size * 4 + 5)) * 0.25
        y2 = np.ones((self.size * 4 + 3, 1)) * 0.25

        hb = np.concatenate(
            (y, self.berries_hl, y, self.berries_hr, y),
            axis=1
        )
        rb = np.concatenate(
            (y, self.berries_rl, y, self.berries_rr, y),
            axis=1
        )

        hc = np.concatenate(
            (y, self.canvas_hl, y, self.canvas_hr, y),
            axis=1
        )
        rc = np.concatenate(
            (y, self.canvas_rl, y, self.canvas_rr, y),
            axis=1
        )

        ss = np.concatenate(
            (x, hb, x, rb, x, hc, x, rc, x),
            axis=0
        )

        nm = self.navi_map
        nm = np.concatenate(
            (y2, nm, y2),
            axis=1
        )
        nm = np.concatenate(
            (x2, nm, x2),
            axis=0
        )

        all = np.concatenate(
            (ss, nm),
            axis=1
        )

        if self.mode == 'revealed':
            score = np.array(self.score_img, dtype=np.float) / 255
            all = np.concatenate(
                (all, score),
                axis=0
            )

        return all

    def reward(self):
        return self.score()

    def exit_condition(self):
        steps = self.steps
        return steps > 10000

    def force_condition(self):
        return False
