from scipy.constants import speed_of_light
from scipy.optimize import minimize, shgo
from scipy import interpolate

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import numpy as np
import skrf as rf

import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# band_frequency = '6-12ghz' # Band of Interest 

mm = 0.001 # перевод мм в метры
const_C = speed_of_light #299792458 #Скорость света в вакуумме
Ghz = 1E9 # множитель для ГГц

MAX_INTERPOLATION_POINTS = 1000
GLOBAL_EPS_BOUND = (1.0006, 6)       # Границы для поиска глобалного минимума eps
GLOBAL_OPTIMIZE_ITERRATIONS = 8

# Задаем длину линии и проницаемость
a = 23*mm
b = 10*mm
L = 100 * mm

L1 = 26.8 * 10 * mm - L
L2 = 0



app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[
                    {'label': 'SParameters', 'value': },
                    {'label': 'Frequency', 'value': },
                    {'label': 'DielectricParam', 'value': },
                    {'label': 'Tangent', 'value': }
                ],   
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[
                    {'label': 'SParameters', 'value': },
                    {'label': 'Frequency', 'value': },
                    {'label': 'DielectricParam', 'value': },
                    {'label': 'Tangent', 'value': }
                ],
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),
])

def Lambda(eps,frq,a):
    # Возвращает длину волны в волноводе,
    # для заданной частоты frq (может быть массивом)
    # волновод с параметрами:
    # шириной a, диэлектрическая проницаемость eps       
    return speed_of_light / frq / (eps - (speed_of_light / frq / 2 / a) ** 2) ** 0.5

def phase_rad(eps, frq, a, L):
    return 2*np.pi*L/Lambda(eps, frq, a)

def phase_grad(eps, frq, a, L):
    return 360*L/Lambda(eps, frq, a)

def theori_dphi(eps, frq, a, L):
    theory_air = phase_grad(1, frq, a, L)
    return phase_grad(eps, frq, a, L) - theory_air

def Calc_delay_SP(Q):
    i = 1j
    #i = complex(0,1)
    Sparam = np.zeros((2, 2), dtype=complex)
    Sparam[0,0] = 0
    Sparam[1,1] = 0
    Sparam[1, 0] = np.exp(-i*Q)
    Sparam[0, 1] = np.exp(-i*Q)
    return Sparam

def Calc_Step_SP(R1,R2):
    Sparam = np.zeros((2, 2), dtype=complex)
    Sparam[0,0] = (R2/R1 - 1)/(R2/R1 + 1)
    Sparam[1,1] = -Sparam[0,0]
    Sparam[1, 0] = 2 * np.sqrt(R2/R1)/(R2/R1 + 1)
    Sparam[0, 1] = 2 * np.sqrt(R2/R1)/(R2/R1 + 1)
    return Sparam

def Calc_R_WaveGuide(a, b, eps, f):
    return 2*b/a*120*np.pi/np.sqrt(eps - (speed_of_light/f/2/a)*(speed_of_light/f/2/a));

def Calc_Line_Phase_Delay(L,eps,frq,a):
    return np.pi * 2 * L / Lambda(eps,frq,a)

def s2t(S):
    """
    Преобразует S-параметры в T-параметры
    """
    tparam = np.zeros((2, 2), dtype=complex)
    tparam[0, 0] = S[0,1] - S[0,0] * S[1,1] / S[1,0]
    tparam[1, 1] = 1 / S[1,0]
    tparam[1, 0] = - S[1,1] / S[1,0]
    tparam[0, 1] = S[0,0] / S[1,0]
    return tparam

def t2s(T):
    """
    Преобразует T-параметры в S-параметры
    """
    S = np.zeros((2, 2), dtype=complex)
    S[0,0] = T[0,1] / T[1,1]
    S[1,0] = 1 / T[1,1]
    S[0,1] = T[0,0] - T[0,1]*T[1,0] / T[1,1]
    S[1,1] = - T[1,0] / T[1,1]
    return S

def attenuation(n, s):
    '''
        n - коэффициент затухания в dB
        s - параметр s21
    '''
    # return 10**(-n / 20 ) * s
    return 10**(-n / 10 ) * s

# Функция извлечения S параметров системы
# (Волновод воздух) - (Скачек сопротивления) - (Волновод диэлектрик) - (Скачек сопротивления) - (Волновод воздух)
# [Sparam] = Calc_WG_Line_Comb_SP(all_eps,a,b,frq,L1,L,L2)
# all_eps = eps - i*eps*tan(d)
# ВИДИМО ОПРЕДЕЛЯЮТ ОБЫЧНО ВСЮ ФУНКЦИЮ tan(d) !!!!!!!!!!!!!!!!!!!!!!!! 
# L1,L2 длина воздушных линий 1 и 2
# a,b - стенки волновода,  frq - частота
# -------------------------------------------------
def Calc_WG_Line_Comb_SP(all_eps, a, b, frq, L1, L, L2):
    # Анализируем с диэлектриком
    # Волновод-воздух
    Sc1 = Calc_delay_SP(Calc_Line_Phase_Delay(L1,1,frq,a))
    Tc1 = s2t(Sc1)
    
    # Неоднородность волновод-воздух волновод-диэлектрик
    Sc2 = Calc_Step_SP(Calc_R_WaveGuide(a,b,1,frq), Calc_R_WaveGuide(a,b,all_eps,frq))
    Tc2 = s2t(Sc2)

    #  Волновод-диэлектрик
    Sc3 = Calc_delay_SP(Calc_Line_Phase_Delay(L,all_eps,frq,a))
    Tc3 = s2t(Sc3)

    # Неоднородность волновод-диэлектрик волновод-воздух
    Sc4 = Calc_Step_SP(Calc_R_WaveGuide(a,b,all_eps,frq), Calc_R_WaveGuide(a,b,1,frq))
    Tc4 = s2t(Sc4)

    # Волновод-воздух
    Sc5 = Calc_delay_SP(Calc_Line_Phase_Delay(L2,1,frq,a))
    Tc5 = s2t(Sc5)

    #return t2s(s2t(Sc1)*s2t(Sc2)*s2t(Sc3)*s2t(Sc4)*s2t(Sc5))
    #return t2s( s2t(Sc1).dot( s2t(Sc2) ).dot( s2t(Sc3) ).dot( s2t(Sc4) ).dot( s2t(Sc5) ) )
    # Sall = t2s( Tc1.dot( Tc2 ).dot( Tc3 ).dot( Tc4 ).dot( Tc5 )) # Без доп. потерь
    Sall = t2s( Tc1.dot( Tc2 ).dot( Tc3 ).dot( Tc4 ).dot( Tc5 )) 
    
    # Добавляем потери
    result = attenuation(0.1, Sall)
    #     result = attenuation(0.00000265, Sall)
    
    return result

def theoretic_s21_mag(eps, tand, frq, a, b, L1, L, L2):
    i = 1j
    #all_eps = eps - i * eps * np.tan(d) # Полная диэлектрическая проницаемость
    all_eps = eps - i * eps * tand # Полная диэлектрическая проницаемость

    Sparam = []
    S21 = []
    #S21_angle = []
    for f in frq:
        x = Calc_WG_Line_Comb_SP(all_eps, a, b, f, L1, L, L2)
        Sparam.append( x )
        S21.append( np.abs(x[0][1]) )
       #S21_angle.append( np.angle(x[0][1], deg=True) )

    S21db = 20 * np.log10(S21)  # Выразили в дБ

    #plt.plot(frq, S21, label = f'EPS={eps}, d={tand}')
    #plt.legend()
    #plt.title('|S21|')    
    return S21

def theoretic_s21(eps, tand, frq, a, b, L1, L, L2):
    i = 1j
    #all_eps = eps - i * eps * np.tan(d) # Полная диэлектрическая проницаемость
    all_eps = eps - i * eps * tand # Полная диэлектрическая проницаемость

    Sparam = []
    S21 = []
    S21_mag = []
    S21_angle = []
    for f in frq:
        x = Calc_WG_Line_Comb_SP(all_eps, a, b, f, L1, L, L2)
        Sparam.append( x )
        S21.append(x[0][1]) # S21 is a complex number
        S21_mag.append( np.abs(x[0][1]) )
        S21_angle.append( np.angle(x[0][1], deg=True) )

  #     S21db = 20 * np.log10(S21)  # Выразили в дБ
  #     S21_angle = np.unwrap(S21_angle)

    return np.abs(S21), S21_angle

def theoretic_s21_deg(eps, tand, frq, a, b, L1, L, L2):
    
    fff = np.linspace(frq[0], frq[-1], MAX_INTERPOLATION_POINTS)    
    mag, angle = theoretic_s21(eps, tand, fff, a, b, L1, L, L2)
    f = interpolate.interp1d(fff, angle)
    return f(frq)


class Air():
    def __init__(self, file_name, name, a, b, L1, L, L2, band_frequency='7-11ghz'):
        """
        file_name - имя файла .s2p
        name - имя, которые будет на графике отображаться например
        L1 - длина левого участка 
        L - длина центрального участка
        L2 - длина правого участка
        """
        # Загружаем данные из файла
        self.data = rf.Network(file_name)
        
        self.name = name
        
        self.band_frequency = band_frequency # Band of Interest 
        
        # параметры волновода
        self.a = a
        self.b = b
        self.L1 = L1
        self.L = L
        self.L2 = L2
        print('L = ', self.L)
        
        #phase = -data[band_frequency].s21.s_deg_unwrap.flatten()
        #
        # Почему тут минус нужно добавить пока не ясно
        # Где-то должно быть ошибка.
        # Надо с этим разобраться.
        #self.phase = -self.data.s21.s_deg_unwrap.flatten()
        
        self.phase = self.data.s21.s_deg_unwrap.flatten()
        
        #self.frq = self.data[band_frequency].frequency.f
        
    def plot_phase(self, band_frequency):
        """
        """
        
        frq = self.get_frq(band_frequency) # band_frequency
        plt.plot(frq / ghz, self.get_phase(band_frequency), label=self.name)
        
    def get_phase(self, band_frequency):
        # Тут меняем знак (не волне понятно почему. Возможно из-за клаибровки)
        #return -self.data[band_frequency].s21.s_deg_unwrap.flatten()
        return self.data[band_frequency].s21.s_deg_unwrap.flatten()
    
    def get_frq(self, band_frequency):
        """
        Возращает точки частот из заданного диапазона,
        для которых есть данные
        """        
        return self.data[band_frequency].frequency.f # band_frequency 
    
    @classmethod
    def _theoretic_s21(cls, eps, tand, frq, a, b, L1, L, L2):
        '''
        Теоретическое значение амплитуды для заданного значения
        eps - диэлетрической проницаемости и
        tand - тангенса угла диэлектрических потерь
        
        возвращет
        амплитуда, фаза
        '''
        i = 1j
        all_eps = eps - i * eps * tand # Полная диэлектрическая проницаемость

        Sparam = []
        S21 = []
        S21_mag = []
        S21_angle = []
        for f in frq:
            x = Calc_WG_Line_Comb_SP(all_eps, a, b, f, 30*mm, L, 30*mm)
            Sparam.append( x )
            S21.append(x[0][1])
            S21_mag.append( np.abs(x[0][1]) )
            S21_angle.append( np.angle(x[0][1], deg=True) )

        # S21db = 20 * np.log10(S21)  # Выразили в дБ
        S21_angle = np.unwrap(S21_angle)
        
        return S21, S21_angle
        
    @classmethod
    def _theoretic_s21_mag(cls, eps, tand, frq, a, b, L1, L, L2):
        '''
        Теоретическое значение амплитуды для заданного значения
        eps - диэлетрической проницаемости и
        tand - тангенса угла диэлектрических потерь        
        
        Возвращает,
        амплитуду
        '''
        
        S21, S21_angle = cls._theoretic_s21(eps, tand, frq, a, b, L1, L, L2)    
        return S21
    
    def ajust_a_l(self):
        """
        Уточняет размеры волновода.
        Возвращает параметр a волновода (широкая стендка) и его длину.
        """
        def f(frq, a, L, L1, L2):
            estimated_deg = phase_grad(1, frq, a, -L1 - L - L2) + 360*2
            diff = air.data[self.band_frequency].s12.s_deg_unwrap.flatten() - (estimated_deg)
        #     plt.plot(frq, diff, label="Разница")
            return np.sum(np.square(diff))

        def f_a(a):
            return f(frq, a, self.L, self.L1, self.L2)

        def f_l(waveguide_length):
            L = self.L
            L2 = self.L2
            L1 = waveguide_length - L - L2            
            return f(frq, self.a, L, L1, L2)

        frq = self.data[self.band_frequency].frequency.f
        
        initial_a = self.a
        res = minimize(f_a, initial_a, method='Nelder-Mead', tol=1e-6)
        self.a = res.x[0]

        initial_waveguide_length = self.L1 + self.L + self.L2
        res = minimize(f_l, initial_waveguide_length, method='Nelder-Mead', tol=1e-6)
        weveguide_length = res.x[0]
 #         self.L1 = weveguide_length - self.L
        
        self.weveguide_length = weveguide_length
        
        print('Уточненный параметр a: ', self.a)
        print('Уточненная длина волновода: ', weveguide_length)
        print('Уточненная длина L1: ', self.L1)
        
        return self.a, weveguide_length
    
    def plot_phase(self):
        """
        Отображает графики измеренной и теоретической фазы
        """
        frq = self.data[self.band_frequency].frequency.f
        
        def find_shift_k():
            def f(k):
                estimated_deg = phase_grad(1, frq, self.a, -self.L1 - self.L - self.L2) + 360*k
                diff = self.data[self.band_frequency].s12.s_deg_unwrap.flatten() - (estimated_deg) 
                return np.sum(np.square(diff))

            res = minimize(f, 1)

            return int( np.round(res.x[0]) )
        
        shift_k = find_shift_k()
        print(shift_k)

        estimated_deg = phase_grad(1, frq, self.a, -self.L1 - self.L - self.L2) + 360*shift_k
        # estimated_deg = np.unwrap(estimated_deg) 

        diff = self.data[self.band_frequency].s12.s_deg_unwrap.flatten() - (estimated_deg) 

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot()
        
 #         plt.figure(figsize=(15,10))
        plt.plot(frq, estimated_deg, label="Модель")
        self.data[self.band_frequency].s12.plot_s_deg_unwrap(linestyle='--', label="Эксперимент")
        plt.title('')
        plt.ylabel('Набег фазы S21, градусы')
        
        # scale_x = Ghz
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/Ghz))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_xlabel("Частота, ГГц")        
        plt.savefig('воздух-воздух')
        
 #         plt.figure(figsize=(15,10))
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot()
        
        plt.plot(frq, diff, label="Вычесленно")
        plt.title('')
        plt.ylabel('Разность фаз, градусы')
        # scale_x = Ghz
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/Ghz))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_xlabel("Частота, ГГц")         
        plt.savefig('воздух-воздух разность фаз')
        

class Dielectric(Air):
    
    def __init__(self, file_name, name, a, b, L1, L, L2, air, band_frequency='7-11ghz'):
        """
        file_name - имя файла .s2p
        name - имя, которые будет на графике отображаться например
        L1 - длина левого участка 
        L - длина центрального участка
        L2 - длина правого участка
        
        air - соответвующая структура, но только с воздухом
        """
        
        super().__init__(file_name, name, a, b, L1, L, L2)
        self.air = air
        self.adjust_a_l() # Уточняем размеры волновода
        
        self.esp = None
        self.tand = None
        
    def find_initial_approximation(self):
        """
        Находим начальное конечное приближение для eps,
        используя глобальную оптимизацию.
        (на основе этого приближение в дальнейшем будет найдено более точное значение)
        """    
        def error_global(eps):
            frq = self.data[self.band_frequency].frequency.f
            tand = 0.0001
            mag, angle = theoretic_s21(eps, tand, frq, self.a, self.b, self.L1, self.L, self.L2)    
            angle = np.array(angle)
            diff = self.data[self.band_frequency].s12.s_deg.flatten() - angle
            return np.sum(np.abs(diff))

        bounds = [GLOBAL_EPS_BOUND]
        result = shgo(error_global, bounds, iters=GLOBAL_OPTIMIZE_ITERRATIONS)
        
        return result.x
    
        
 #     @lru_cache()
    def find_params(self, band_frequency):
        """ 
            Находит параметры диэлектрика
            Используется модель
            (Волновод воздух) - (Скачек сопротивления) - (Волновод диэлектрик) - (Скачек сопротивления) - (Волновод воздух)

            band_frequency - диапазон частот

            возвращает:
            eps - диэлектрическая проницаемость
            tand - тангенс угла потерь

        """

        def fun(eps, tand, sample_data, frq):

            frq = self.data[band_frequency].frequency.f
            theoretic = theoretic_s21_mag(eps, tand, frq, self.a, self.b, self.L1, self.L, self.L2)

            #return np.sum(np.square(theoretic + const - sample_data))
            return np.sum(np.square(theoretic - sample_data))

        def func(x):
            return fun(*x, sample_data, frq)

        def func_to_find_const(*x):
            eps, const, sample_data, frq = x
            return fun(eps, const, sample_data, frq)

        frq = self.get_frq(band_frequency)
        sample_data = self.data[band_frequency].s21.s_mag.flatten()
        
        initial_eps = self.find_initial_approximation()[0]

 #         x0 = (2.6, 0.002) # Начальное приближение
        x0 = (initial_eps, 0.002) # Начальное приближение
        bnds = ((1.0006, 10), (-0.01, 0.01))

        res = minimize(func, x0, bounds=bnds)
        eps = res.x[0]
        tand = res.x[1]
        
        return eps, tand

 #_____________________________

    def get_dphi(self, band_frequency):
        return self.get_phase(band_frequency) - self.air.get_phase(band_frequency)
    
    def find_eps(self, band_frequency):
        """ 
            Старая функция.!!!
            Находит значение eps 
            
            Используется разность фаз 
            между волноводном с диэлектриком и волноводом с воздухом
                        
            
            band_frequency - диапазон частот
            
            возвращает:
            eps - диэлектрическая проницаемость
            const - константа
            
        """

        def fun(eps, const, sample_data, frq):
            
            theoretic = theori_dphi(eps, frq, self.a, self.L)
            return np.sum(np.square(theoretic + const - sample_data))

        def func(x):
            return fun(*x, sample_data, frq)
        
        def func_to_find_const(*x):
            eps, const, sample_data, frq = x
            return fun(eps, const, sample_data, frq)
        
        frq = self.get_frq(band_frequency)
        sample_data = self.get_dphi(band_frequency)

        x0 = (2.5, -100) # Начальное приближение
        bnds = ((1.5, 40), (None, None))

        res = minimize(func, x0, bounds=bnds)
        eps = res.x[0]
        const = res.x[1]      

        return eps, const
    
    def get_difference_db(self, band_frequency, other_sample):
        """ Сравнивает два диэлектрика по S21 в db """
        diff_db = self.data[band_frequency].s21.s_db.flatten() - other_sample.data[band_frequency].s21.s_db.flatten()
        return diff_db
    
    def get_rid_of_air(self, band_frequency):
        diff = self.get_difference_db(band_frequency, self.air)
        return diff

    def get_difference_db_wo_air(self, band_frequency, other_sample):
        """ Сравнивает два диэлектрика по S21 в db,
        предварительно вычитается воздух """
        #diff_wo_air = (self.air.data[band_frequency] / self.data[band_frequency]).s21.s_db.flatten()
        
        diff = self.get_rid_of_air(band_frequency) - other_sample.get_rid_of_air(band_frequency)
        return diff
    
    
    def plot_dielectric_params(self, band_frequency):
        
        frq = self.get_frq(band_frequency)
        sample_data = self.get_dphi(band_frequency)

        eps, const = self.find_eps(band_frequency)

        plt.plot(frq / ghz, theori_dphi(eps, frq, self.a, self.L) + const, '--b', label=f"Теоретическое значение, $\epsilon = {eps:.3f}$, const = {const:.2f}")
        plt.plot(frq / ghz, sample_data, 'r', label='Эксперимент')
        
        
        frequency_for_title = band_frequency.replace('ghz', 'ГГц') # Исправляем для отображения ГГц
        title = self.name + ', ' + frequency_for_title

        plt.title(title)
        plt.xlabel('Частота, ГГц')
        #plt.ylabel('Градусы')
        plt.legend()
        
    def adjust_a_l(self):
        """
        Уточняет размеры волновода.
        """
        self.air.ajust_a_l()
        
        self.a = self.air.a
        self.L1 = self.air.weveguide_length - self.L 
        
        print('Уточненный параметр a: ', self.a)
        print('Уточненная длина волновода: ', self.L + self.L1 + self.L2)
        print('Уточненная длина L1: ', self.L1)        
        print('L = ', self.L)
        


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    data_dir = r'\Users\User\Desktop\finally/'

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              Output('indicator-graphic', 'figure'),
              Input('upload-data', 'contents'),
              Input('xaxis-column', 'value'),
              Input('yaxis-column', 'value'),
              Input('xaxis-type', 'value'),
              Input('yaxis-type', 'value'),
              State('upload-data', 'last_modified'),
              State('upload-data', 'filename'),
            )

def multi_output(value):
    if value is None:
        raise PreventUpdate

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):

    fig = px.line(x=df[df['Indicator Name'] == xaxis_column_name]['Value'],
                     y=df[df['Indicator Name'] == yaxis_column_name]['Value'],)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return fig        



if __name__ == '__main__':
    app.run_server(debug=True)