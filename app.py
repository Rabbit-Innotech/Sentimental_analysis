





import os
import re
import io
import random
import warnings
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import requests

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

class MultiLevelSentimentAnalyzer:
    """Enhanced multi-level sentiment analyzer with emotion, topic, and behavior analysis"""

    def __init__(self):
        # Sentiment keywords
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'awesome',
            'perfect', 'happy', 'pleased', 'satisfied', 'brilliant', 'smart', 'helpful', 'better', 'improved', 'outstanding',
            'superb', 'remarkable', 'impressive', 'beneficial', 'effective', 'efficient', 'convenient', 'reasonable',
            'incredible', 'beautiful', 'cool', 'fun', 'nice', 'positive', 'respectful', 'easy', 'kind', 'lovely',
            'supportive', 'encouraging', 'valuable', 'trustworthy', 'responsive', 'friendly', 'clean', 'organized', 'creative',
            'motivating', 'life-changing', 'strong', 'genuine', 'enjoyable', 'welcoming', 'fascinating', 'legendary',
            'unforgettable', 'smooth', 'safe', 'secure', 'trusted', 'polite', 'cheerful', 'passionate', 'inspirational',
            'awesome', 'mind-blowing', 'game-changer', 'funny', 'laughable', 'uplifting', 'vibrant', 'beautifully', 'fast',
            'affordable', 'accessible', 'responsive', 'insightful', 'well-done', 'enlightening', 'grateful', 'talented',
            'professional', 'innovative', 'wise', 'superior', 'excellent', 'charismatic', 'generous', 'caring', 'cute',
            'admirable', 'meaningful', 'dedicated', 'memorable', 'trusting', 'passion', 'respect', 'hopeful', 'amused',
            'byiza', 'meza', 'cyiza', 'ntagereranywa', 'bitangaje', 'birashimishije', 'nkunda', 'nishimiye', 'neza',
            'ibitangaza', 'birakwiye', 'uburanga', 'amahoro', 'ishimwe', 'ibyishimo', 'gukunda', 'amahire', 'amahirwe',
            'gutera imbere', 'gukora neza', 'ibikwiye', 'umumaro', 'ingirakamaro', 'ubunyamwuga', 'umuhanga', 'gufasha',
            'kumwenyura', 'gushima', 'gukemura', 'gukorana', 'ubushobozi', 'ubwitange', 'gukundwa', 'gukundisha',
            'umunezero', 'ibyiza', 'urukundo', 'ubuntu', 'ubwenge', 'gukora', 'guha agaciro', 'agaciro', 'umusaruro',
            'guhiga abandi', 'gusetsa', 'gutanga', 'gushimira', 'umutekano', 'ubushuti', 'guhuza', 'gusangira', 'gusabana',
            'kwemera', 'kwishimira', 'gukorana neza', 'ubufasha', 'gukora akazi neza', 'ubupfura', 'gukomera', 'gukomeza',
            'gutera imbere', 'ubwitonzi', 'ikinyabupfura', 'ubugwaneza', 'umucyo', 'gusobanukirwa', 'guharanira iterambere',
            'gukemura ibibazo', 'urukundo', 'umucyo', 'gufasha abandi', 'kubaha', 'kwihangana', 'gukomeza gukora',
            'kwiyubaka', 'kwihangira', 'guha abandi icyizere', 'kuba inyangamugayo', 'gusabana neza', 'gufatanya',
            'kwizera', 'gukora neza cyane', 'kurushaho', 'guhitamo neza', 'gukomeza inzira', 'ubutwari', 'ubumuntu',
            'kubaha abandi', 'umuhate', 'icyizere', 'ubumwe', 'gushimangira', 'gukora icyiza', 'gukomeza gukunda',
            'ubutwari', 'gukomera ku ntego', 'ubudasa', 'ubushishozi', 'intambwe', 'gushimangira icyiza'

        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'angry', 'frustrated', 'confused',
            'expensive', 'unfair', 'discriminate', 'wrong', 'problem', 'issue', 'difficult', 'complicated', 'poor', 'fails',
            'useless', 'broken', 'slow', 'unreliable', 'inadequate', 'insufficient', 'disgusting', 'annoying', 'boring',
            'sad', 'waste', 'fake', 'scam', 'toxic', 'stressful', 'lazy', 'messy', 'offensive', 'rude', 'unhelpful',
            'jealous', 'biased', 'neglect', 'terrified', 'ashamed', 'overpriced', 'insecure', 'painful', 'negative',
            'ignored', 'blocked', 'weird', 'unprofessional', 'unacceptable', 'untrustworthy', 'lost', 'poorly', 'dumb',
            'unhappy', 'unpleasant', 'annoyed', 'abandoned', 'noisy', 'nonsense', 'misleading', 'immature', 'unnecessary',
            'repetitive', 'pointless', 'angrily', 'resentful', 'depressed', 'unsatisfied', 'unjust', 'illogical', 'absurd',
            'ridiculous', 'cringe', 'awkward', 'fail', 'brokenness', 'overwhelmed', 'ineffective', 'difficulties',
            'uncooperative', 'lackluster', 'gross', 'disturbing', 'inappropriate', 'delayed', 'buggy', 'flawed', 'crash',
            'annoyance', 'drama', 'uselessness', 'unusable', 'terrifying', 'obnoxious',
            'bibi', 'bibi cyane', 'sinakunze', 'mbabajwe', 'gukomeretsa', 'gukara', 'gusebanya', 'urwango', 'uburakari',
            'gutuka', 'gucira abandi urubanza', 'gusebya', 'gucika intege', 'umunabi', 'guhomba', 'ikibazo', 'ibibazo',
            'uruhare ruke', 'gutererana', 'ubusambo', 'kubeshya', 'ibinyoma', 'ugusebanya', 'guciraho iteka',
            'gukomerera', 'inzangano', 'gutinda', 'guhutaza', 'kwirengagiza', 'guharira', 'kwanga', 'kurakara',
            'gucika intege', 'gutuka cyane', 'kubura icyizere', 'kubura igisubizo', 'nta gaciro', 'ntacyo bimaze',
            'nta kamaro', 'ikimwaro', 'kwishinja', 'gutinya', 'gutinda', 'kwicwa n’agahinda', 'kwanga ibintu byose',
            'kutumvikana', 'ubukene', 'ibibazo byinshi', 'gukena', 'gushavura', 'gutwikwa', 'kuribwa', 'umujinya',
            'uburakari bwinshi', 'gukomerwa', 'kwangirika', 'gushwana', 'gutandukana nabi', 'gufatwa nabi', 'guhunga',
            'kugira ubwoba', 'gufata nabi', 'kwigomeka', 'kwibagirana', 'kudashimwa', 'kuba wenyine', 'kudasobanura',
            'kutumva', 'kudakora', 'kudashima', 'kudashobora', 'kwibaza cyane', 'gucika intege', 'kubura amahoro',
            'uburwayi', 'indwara', 'kwanga abandi', 'kugira ipfunwe', 'gusaza nabi', 'gusenya', 'guhagarika', 'gukandamizwa',
            'gucibwa', 'gusuzugurwa', 'uburakari bwinshi', 'kutihanganira', 'guhorana agahinda', 'kugabanuka',
            'kubura uburenganzira', 'guhora urira', 'kubabara', 'kwibona nabi', 'gusenyuka', 'kurakarira abandi',
            'gukoresha nabi', 'kugira ishyari', 'guteshwa agaciro', 'kubura amahirwe'

        }

        # Emotion mapping
        self.emotion_keywords = {
              'joy': [
                  # English
                  'amazing', 'fantastic', 'wonderful', 'love', 'perfect', 'awesome', 'outstanding', 'happy', 'excited', 'thrilled',
                  'delight', 'bliss', 'cheerful', 'ecstatic', 'elated', 'euphoric', 'gleeful', 'jubilant', 'joyful', 'merry',
                  'overjoyed', 'pleased', 'radiant', 'satisfied', 'smiling', 'sunny', 'upbeat',
                  # Kinyarwanda
                  'ibyishimo', 'ishimwe', 'gushimishwa', 'gukunda', 'gukundwa', 'gunezerwa', 'guseka', 'gukanyurwa', 'gukanyurwa cyane', 'gukanyurwa bikabije'
              ],
              'gratitude': [
                  # English
                  'thanks', 'thank', 'grateful', 'appreciate', 'appreciated', 'blessing', 'thankful', 'gratitude', 'acknowledge', 'recognize',
                  # Kinyarwanda
                  'urakoze', 'murakoze', 'ndabashimira', 'ndashimira', 'ishimwe', 'gushima', 'gushimira', 'gukenguruka', 'gukenguruka cyane'
              ],
              'excitement': [
                  # English
                  'excited', 'thrilled', 'delighted', 'amazing', 'incredible', 'wow', 'ecstatic', 'elated', 'eager', 'enthusiastic', 'overjoyed', 'buzzing',
                  # Kinyarwanda
                  'gushimishwa', 'gukanyurwa', 'gukanyurwa cyane', 'gukanyurwa bikabije', 'gukanyurwa byimazeyo', 'gukanyurwa cyane cyane'
              ],
              'anger': [
                  # English
                  'awful', 'terrible', 'hate', 'worst', 'horrible', 'angry', 'furious', 'mad', 'irritated', 'annoyed', 'outraged', 'resentful', 'enraged',
                  # Kinyarwanda
                  'umujinya', 'kurakara', 'gukara', 'gukara cyane', 'gukara bikabije', 'gukara byimazeyo', 'gukara cyane cyane'
              ],
              'sadness': [
                  # English
                  'regret', 'disappointed', 'sad', 'upset', 'depressed', 'crying', 'unhappy', 'melancholy', 'sorrowful', 'gloomy', 'heartbroken', 'mournful',
                  # Kinyarwanda
                  'agahinda', 'kubabara', 'kubabara cyane', 'kubabara bikabije', 'kubabara byimazeyo', 'kubabara cyane cyane'
              ],
              'frustration': [
                  # English
                  'frustrating', 'annoying', 'irritating', 'confusing', 'complicated', 'exasperating', 'discouraging', 'disheartening', 'aggravating',
                  # Kinyarwanda
                  'kugira umujinya', 'kugira umujinya mwinshi', 'kugira umujinya mwinshi cyane', 'kugira umujinya mwinshi cyane cyane'
              ],
              'concern': [
                  # English
                  'worried', 'concerned', 'anxious', 'uncomfortable', 'unsure', 'nervous', 'apprehensive', 'uneasy', 'troubled', 'distressed',
                  # Kinyarwanda
                  'impungenge', 'guhangayika', 'guhangayika cyane', 'guhangayika bikabije', 'guhangayika byimazeyo', 'guhangayika cyane cyane'
              ],
              'surprise': [
                  # English
                  'surprising', 'shocked', 'unexpected', 'unbelievable', 'wow', 'astonished', 'amazed', 'startled', 'stunned', 'speechless',
                  # Kinyarwanda
                  'gutungurwa', 'gutungurwa cyane', 'gutungurwa bikabije', 'gutungurwa byimazeyo', 'gutungurwa cyane cyane'
              ],
              'fear': [
                  # English
                  'scared', 'afraid', 'terrified', 'worried', 'nervous', 'anxious', 'panicked', 'frightened', 'alarmed', 'apprehensive',
                  # Kinyarwanda
                  'ubwoba', 'gutinya', 'gutinya cyane', 'gutinya bikabije', 'gutinya byimazeyo', 'gutinya cyane cyane'
              ]
          }

        # Topic mapping
        self.topic_keywords = {
            'product': [
                # English
                'product', 'item', 'purchase', 'buy', 'bought', 'quality', 'build', 'material', 'design', 'goods', 'brand', 'model', 'features',
                # Kinyarwanda
                'igicuruzwa', 'igikoresho', 'gucuruza', 'kugura', 'naguriye', 'ubwiza', 'imiterere', 'ibikoresho', 'ikoranabuhanga', 'ubuziranenge'
            ],
            'delivery': [
                # English
                'delivery', 'shipping', 'arrived', 'package', 'transport', 'logistics', 'dispatched', 'courier', 'received', 'on time', 'delivered',
                # Kinyarwanda
                'gutanga', 'kohereza', 'paki', 'yoherejwe', 'yakiriwe', 'itarageze', 'itinda', 'ku gihe', 'ibikoresho byageze', 'gutwarwa', 'ubwikorezi'
            ],
            'service': [
                # English
                'service', 'support', 'help', 'staff', 'team', 'customer service', 'assistance', 'agent', 'representative', 'guidance', 'consulting',
                # Kinyarwanda
                'serivisi', 'ubufasha', 'gufasha', 'abakozi', 'itsinda', 'serivisi z’abakiriya', 'ubujyanama', 'ubuyobozi', 'ubufatanye'
            ],
            'pricing': [
                # English
                'price', 'cost', 'expensive', 'cheap', 'value', 'money', 'budget', 'affordable', 'worth', 'discount', 'sale', 'offer', 'deal',
                # Kinyarwanda
                'igiciro', 'amahera', 'kugura', 'guhenda', 'guhendutse', 'ubushobozi', 'igiciro cyiza', 'ifatira ku giciro', 'ingano y’amafaranga'
            ],
            'technology': [
                # English
                'website', 'app', 'interface', 'login', 'account', 'software', 'system', 'platform', 'technology', 'bug', 'crash', 'update', 'feature',
                # Kinyarwanda
                'urubuga', 'porogaramu', 'konti', 'kwinjira', 'gukoresha', 'ikoranabuhanga', 'sisitemu', 'uburyo bwo gukoresha', 'udushya', 'gusubizamo'
            ],
            'returns': [
                # English
                'return', 'refund', 'exchange', 'policy', 'warranty', 'return policy', 'replacement', 'back', 'sent back', 'money back',
                # Kinyarwanda
                'gusubiza', 'kwishyurwa', 'gusimbuza', 'politiki y’inyemezabwishyu', 'garanti', 'ubwishingizi', 'gusaba amafaranga asubizwa'
            ],
            'experience': [
                # English
                'experience', 'overall', 'general', 'impression', 'feeling', 'satisfaction', 'feedback', 'opinion', 'review', 'thoughts', 'enjoyed',
                # Kinyarwanda
                'uburambe', 'ubunararibonye', 'ibyiyumviro', 'ibitekerezo', 'isura rusange', 'ibyo nabonye', 'uburyo numvise', 'gunezerwa', 'kumva neza'
            ],
            'content': [
                # English
                'video', 'content', 'information', 'educational', 'tutorial', 'explanation', 'blog', 'post', 'article', 'lecture', 'training',
                # Kinyarwanda
                'videwo', 'ibikubiyeho', 'amakuru', 'ubumenyi', 'kwigisha', 'ubusobanuro', 'inyigisho', 'amasomo', 'inkuru', 'inyandiko'
            ],
            'performance': [
                # English
                'performance', 'speed', 'efficiency', 'effectiveness', 'results', 'function', 'response time', 'work', 'output', 'operation', 'benchmark',
                # Kinyarwanda
                'imikorere', 'umuvuduko', 'ubushobozi', 'gukora neza', 'ibisubizo', 'ibyavuyemo', 'akazi', 'umusaruro', 'gutanga ibisubizo', 'ubwiza bw’imikorere'
            ]
        }

        # Behavior patterns
        self.behavior_indicators = {
            'promoting': [
                # English
                'recommend', 'suggest', 'share', 'tell others', 'spread word', 'promote', 'advocate', 'endorse', 'encourage',
                # Kinyarwanda
                'gusaba', 'gusaba abandi', 'gusangira', 'kubwira abandi', 'gukangurira', 'gushishikariza', 'kumenyekanisha', 'gushyigikira', 'guhamagarira'
            ],
            'complaining': [
                # English
                'complain', 'report', 'issue', 'problem', 'dissatisfied', 'frustrated', 'disappointed', 'criticize', 'blame', 'file a complaint',
                # Kinyarwanda
                'kwinubira', 'gutanga ikirego', 'ikibazo', 'ikibazo cyagaragaye', 'ntibanyuzwe', 'kurakara', 'kunenga', 'gucira urubanza', 'gutinyuka kuvuga', 'gutanga igitekerezo kibi'
            ],
            'supporting': [
                # English
                'support', 'defend', 'backing', 'endorsing', 'agree with', 'stand with', 'uplift', 'encourage', 'stand behind',
                # Kinyarwanda
                'gushyigikira', 'kurengera', 'gufasha', 'kubogamira', 'kwemera', 'guharanira', 'kuba inyuma ye', 'kwifatanya', 'kuba ku ruhande rwabo'
            ],
            'questioning': [
                # English
                'question', 'wonder', 'curious', 'how', 'why', 'what', 'inquire', 'ask', 'confused', 'not sure',
                # Kinyarwanda
                'kubaza', 'gutekereza', 'kwibaza', 'kugira amatsiko', 'kuki', 'ni iki', 'ese', 'biragenda bite', 'ntabwo nsobanukiwe', 'sinzi'
            ],
            'comparing': [
                # English
                'compare', 'versus', 'better than', 'worse than', 'similar to', 'different from', 'more than', 'less than', 'side by side', 'evaluation',
                # Kinyarwanda
                'gereranya', 'kurusha', 'kurusha abandi', 'ku rwego rwo hasi', 'usa na', 'utandukanye na', 'kuruta', 'gusa na', 'kwigereranya', 'gupima'
            ]
        }

        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can', 'not', 'now',
            'then', 'when', 'where', 'why', 'how', 'so', 'if', 'because', 'as', 'about', 'from', 'up', 'down', 'out',
            'over', 'under', 'again', 'more', 'most', 'some', 'any', 'each', 'few', 'all', 'no', 'nor', 'only', 'own',
            'same', 'such', 'very', 'too', 'just', 'also', 'even', 'still', 'yet', 'ever', 'never', 'get', 'make', 'take',
            'go', 'come', 'know', 'see', 'think', 'say', 'tell', 'ask', 'want', 'need', 'let', 'do', 'did', 'does', 'am',
            'you are', 'we', 'our', 'us', 'they', 'them', 'their', 'me', 'my', 'mine', 'your', 'his', 'her', 'its',
            'i', 'it', 'he', 'she', 'what', 'which', 'who', 'whom', 'shall',
            'na', 'kandi', 'cyangwa', 'ariko', 'mu', 'ku', 'ya', 'yo', 'bya', 'iyo', 'uko', 'muri', 'kugira', 'ngo',
            'bwo', 'ari', 'bari', 'yari', 'ntabwo', 'ntacyo', 'hari', 'harimo', 'hagiye', 'cyane', 'ubwo', 'maze',
            'noneho', 'nuko', 'ubundi', 'kuko', 'ubwo', 'icyo', 'mbese', 'yawe', 'nk’uko', 'mbere', 'nyuma', 'ariko',
            'ni', 'ibi', 'iyo', 'uyu', 'ubwo', 'iki', 'ubwo', 'turi', 'sinzi', 'kugirango', 'ubwo', 'niba', 'ubwo',
            'kera', 'gusa', 'nawe', 'ntacyo', 'kuko', 'ibyo', 'twese', 'cyane', 'ushaka', 'avuga', 'ntabwo', 'urimo',
            'turi', 'muri', 'wowe', 'ntacyo', 'twagiye', 'hariya', 'nimba', 'ubwo', 'mu gihe', 'ko', 'niyo', 'mwese',
            'kuba', 'uko', 'ubwo', 'kugeza', 'aho', 'nka', 'benshi', 'bake', 'abantu', 'umuntu', 'ubwo', 'niyo',
            'naho', 'ushobora', 'ushaka', 'kuri', 'ibyo byose', 'aho', 'uwari', 'kuva', 'ushobora', 'icyo gihe',
            'nuko', 'kugira ngo', 'ubwo', 'ya', 'ubu', 'ubwo', 'ntacyo bitwaye'

        }

    def analyze_sentiment(self, text):
        """Basic sentiment analysis"""
        if not text or len(str(text).strip()) == 0:
            return "Neutral", 0.5

        text_lower = str(text).lower()

        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)

        # Check for mixed sentiment indicators
        mixed_indicators = [
                    # English
                    'but', 'however', 'although', 'though', 'except', 'while', 'nevertheless', 'nonetheless', 'even though', 'on the other hand',
                    'yet', 'still', 'despite', 'in contrast', 'instead',

                    # Kinyarwanda
                    'ariko', 'nyamara', 'nubwo', 'ubwo', 'keretse', 'mu gihe', 'ubwo nubwo', 'ubwo ariko', 'ubwo nubwo byari bimeze bityo',
                    'nubwo byari bimeze bityo', 'nyamara nubwo', 'kandi', 'ariko se', 'gusa', 'icyakora', 'aho kugira ngo'
                ]

        has_mixed = any(indicator in text_lower for indicator in mixed_indicators)

        if has_mixed and pos_count > 0 and neg_count > 0:
            confidence = min(0.8, 0.6 + abs(pos_count - neg_count) * 0.05)
            return "Mixed", confidence
        elif pos_count > neg_count:
            confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
            return "Positive", confidence
        elif neg_count > pos_count:
            confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
            return "Negative", confidence
        else:
            return "Neutral", 0.5

    def analyze_emotion(self, text, sentiment):
        """Analyze emotions in text"""
        if not text:
            return "Neutral"

        text_lower = str(text).lower()
        detected_emotions = []

        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion.title())

        # Handle mixed sentiment emotions
        if sentiment == "Mixed":
            positive_emotions = [
                      # English
                      'Joy', 'Gratitude', 'Excitement', 'Happiness', 'Love', 'Satisfaction', 'Relief', 'Contentment', 'Pride', 'Hope',
                      'Delight', 'Admiration', 'Peace', 'Confidence', 'Appreciation', 'Cheerfulness', 'Optimism',

                      # Kinyarwanda
                      'ibyishimo',        # Joy
                      'ishimwe',          # Gratitude
                      'ibyishimo byinshi',# Excitement
                      'umunezero',        # Happiness
                      'urukundo',         # Love
                      'kunezerwa',        # Satisfaction
                      'kuruhuka umutima', # Relief
                      'gutekana',         # Contentment / Peace
                      'kwirata',          # Pride
                      'icyizere',         # Hope
                      'gushimishwa',      # Delight
                      'gukunda',          # Admiration
                      'amahoro',          # Peace
                      'kwizera',          # Confidence
                      'gushimira',        # Appreciation
                      'guseka',           # Cheerfulness
                      'kugira icyizere'   # Optimism
                  ]

            negative_emotions = [
                      # English
                      'Anger', 'Sadness', 'Frustration', 'Fear', 'Disappointment', 'Worry', 'Anxiety', 'Hopelessness', 'Loneliness',
                      'Shame', 'Guilt', 'Bitterness', 'Irritation', 'Depression', 'Grief', 'Stress', 'Envy', 'Resentment', 'Hurt',
                      'Discouragement', 'Despair', 'Embarrassment', 'Tension', 'Regret',

                      # Kinyarwanda
                      'uburakari',        # Anger
                      'agahinda',         # Sadness
                      'kugira umujinya',  # Frustration
                      'ubwoba',           # Fear
                      'kwiheba',          # Hopelessness
                      'gucika intege',    # Discouragement
                      'kwiheba cyane',     # Despair
                      'kwicuza',          # Regret
                      'kwigunga',         # Loneliness
                      'isoni',            # Shame
                      'ikimwaro',         # Embarrassment
                      'icyaha',           # Guilt
                      'gucika intege',    # Disappointment / Discouragement
                      'ishyari',          # Envy
                      'urwango',          # Bitterness / Resentment
                      'uburwayi bwo mu mutwe', # Depression
                      'umuhangayiko',     # Anxiety / Stress
                      'umubabaro',        # Grief / Hurt
                      'irungu',           # Loneliness / Sadness
                      'umushiha',         # Irritation / Anger
                      'kubura icyizere',  # Hopelessness
                      'gufatwa nabi'      # Hurt / Feeling mistreated
                  ]


            pos_emotions = [e for e in detected_emotions if e in positive_emotions]
            neg_emotions = [e for e in detected_emotions if e in negative_emotions]

            if pos_emotions and neg_emotions:
                return f"{pos_emotions[0]} (positive aspect), {neg_emotions[0]} (negative aspect)"

        if detected_emotions:
            return detected_emotions[0]

        # Fallback based on sentiment
        emotion_map = {
            'Positive': 'Joy',
            'Negative': 'Sadness',
            'Mixed': 'Ambivalence',
            'Neutral': 'Neutral'
        }
        return emotion_map.get(sentiment, 'Neutral')

    def analyze_topic(self, text):
        """Analyze topics/themes in text"""
        if not text:
            return "General"

        text_lower = str(text).lower()
        detected_topics = []

        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic.title())

        return ", ".join(detected_topics) if detected_topics else "General"

    def analyze_behavior(self, text, sentiment, emotion, topic):
        """Analyze behavioral patterns"""
        if not text:
            return "Neutral"

        text_lower = str(text).lower()

        # Check for specific behavior indicators
        for behavior, keywords in self.behavior_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                return behavior.title()

        # Behavior inference based on sentiment and emotion
        if sentiment == "Mixed":
            if "Product" in topic and "Delivery" in topic:
                return "Loyal with concern"
            return "Ambivalent"

        if sentiment == "Positive":
            if emotion == "Gratitude":
                return "Supportive"
            if "Service" in topic:
                return "Appreciative"
            return "Promoting"

        if sentiment == "Negative":
            if emotion in ["Sadness", "Regret"]:
                return "Rejecting"
            if emotion == "Anger":
                return "Complaining"
            if "Quality" in topic:
                return "Critical"
            return "Dissatisfied"

        return "Neutral"

    def multi_level_analysis(self, text):
        """Perform complete multi-level analysis"""
        sentiment, confidence = self.analyze_sentiment(text)
        emotion = self.analyze_emotion(text, sentiment)
        topic = self.analyze_topic(text)
        behavior = self.analyze_behavior(text, sentiment, emotion, topic)

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion,
            'topic': topic,
            'behavior': behavior
        }

    def extract_keywords(self, text, top_n=3):
        """Extract key words from text"""
        if not text:
            return []

        words = re.findall(r'\b[a-zA-Z]{3,15}\b', str(text).lower())
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 3]
        word_counts = Counter(filtered_words)
        top_keywords = word_counts.most_common(top_n)

        return [word for word, count in top_keywords]

class EnhancedDataProcessor:
    """Enhanced data processor with multi-level analysis"""

    def __init__(self):
        self.analyzer = MultiLevelSentimentAnalyzer()
        self.data = None
        self.default_comments = [
            "This new distance fare is really fair. I pay less for short trips!",
            "It's confusing, I don't know how much I'll pay now.",
            "RURA should have informed us better about this change.",
            "Good step towards fairness and modernization.",
            "Too expensive now! I hate this new system.",
            "The distance-based system makes so much more sense than flat rates.",
            "Finally a fair system — short-distance commuters benefit the most!",
            "I'm still unsure how the new rates are calculated. Needs clarity.",
            "Smart move toward a fairer system, but more awareness is needed.",
            "I'm paying more now and it feels unjust.",
            "Great initiative but poor implementation.",
            "Now I know exactly what I'm paying for. Transparent and fair.",
            "The fare calculator is very helpful.",
            "I've noticed faster service since the new system launched.",
            "Distance-based fares are the future of transportation.",
            "This discriminates against people living in rural areas!",
            "My transportation costs have decreased by 30%!",
            "Very impressed with the new fare calculation technology.",
            "Love how I can now predict exactly what my trip will cost.",
            "Works well in urban areas but rural commuters are suffering."
        ]
    def generate_sample_data(self):
        """Generate sample dataset with multi-level analysis"""
        data = []
        base_time = datetime.now() - timedelta(hours=48)

        for i, comment in enumerate(self.default_comments):
            timestamp = base_time + timedelta(hours=random.uniform(0, 48))

            # Perform multi-level analysis
            analysis = self.analyzer.multi_level_analysis(comment)
            keywords = self.analyzer.extract_keywords(comment, 3)
            keyword_str = ", ".join(keywords) if keywords else "N/A"

            data.append({
                "Datetime": timestamp,
                "Text": comment,
                "Sentiment": analysis['sentiment'],
                "Confidence": round(analysis['confidence'], 3),
                "Emotion": analysis['emotion'],
                "Topic": analysis['topic'],
                "Behavior": analysis['behavior'],
                "Keywords": keyword_str
            })

        self.data = pd.DataFrame(data)
        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
        return self.data

    def process_file(self, file):
        """Process uploaded CSV/Excel file with multi-level analysis"""
        if file is None:
            return self.generate_sample_data()

        try:
            # Read file based on extension
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name)
            else:
                return self.generate_sample_data()

            # Check for required column
            if 'Text' not in df.columns:
                raise ValueError("File must contain a 'Text' column")

            # Process data with multi-level analysis
            processed_data = []
            for idx, row in df.iterrows():
                text = str(row['Text']) if pd.notna(row['Text']) else ""

                # Handle timestamp
                if 'Datetime' in df.columns and pd.notna(row['Datetime']):
                    timestamp = pd.to_datetime(row['Datetime'])
                else:
                    timestamp = datetime.now() - timedelta(hours=len(df)-idx)

                # Perform multi-level analysis
                analysis = self.analyzer.multi_level_analysis(text)
                keywords = self.analyzer.extract_keywords(text, 3)
                keyword_str = ", ".join(keywords) if keywords else "N/A"

                processed_data.append({
                    "Datetime": timestamp,
                    "Text": text,
                    "Sentiment": analysis['sentiment'],
                    "Confidence": analysis['confidence'],
                    "Emotion": analysis['emotion'],
                    "Topic": analysis['topic'],
                    "Behavior": analysis['behavior'],
                    "Keywords": keyword_str
                })

            self.data = pd.DataFrame(processed_data)
            return self.data

        except Exception as e:
            print(f"Error processing file: {e}")
            return self.generate_sample_data()

    def extract_video_id(self, video_url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        return None

    def process_youtube_comments(self, video_url):
        """Process YouTube video comments with multi-level analysis"""
        api_key = "AIzaSyDlGn2abWfnPLb5JL2e9H7MrujvEDuBHtI"

        # Validate YouTube URL
        youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
        if not re.match(youtube_pattern, video_url):
            raise gr.Error("Please provide a valid YouTube video link.")

        # Extract video ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise gr.Error("Could not extract video ID from URL.")

        try:
            # Fetch comments using YouTube API
            comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'key': api_key,
                'maxResults': 1000,  # Limit for demo purposes
                'order': 'time'
            }

            all_comments = []
            next_page_token = None

            while len(all_comments) < 5000:  # Limit total comments
                if next_page_token:
                    params['pageToken'] = next_page_token

                response = requests.get(comments_url, params=params)

                if response.status_code != 200:
                    raise gr.Error(f"API request failed: {response.status_code} - {response.text}")

                data = response.json()

                if 'items' not in data or not data['items']:
                    break

                for item in data['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'text': comment['textDisplay'],
                        'time': comment['publishedAt'],
                        'author': comment['authorDisplayName'],
                        'likes': comment['likeCount']
                    })

                # Check if there are more pages
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break

            if not all_comments:
                raise gr.Error("No comments found for this video.")

            processed_data = []
            for comment in all_comments:
                text = comment.get('text', '')
                # Convert ISO timestamp to datetime
                timestamp = datetime.fromisoformat(comment.get('time', '').replace('Z', '+00:00'))

                # Perform multi-level analysis
                analysis = self.analyzer.multi_level_analysis(text)
                keywords = self.analyzer.extract_keywords(text, 3)
                keyword_str = ", ".join(keywords) if keywords else "N/A"

                processed_data.append({
                    "Datetime": timestamp,
                    "Text": text,
                    "Sentiment": analysis['sentiment'],
                    "Confidence": analysis['confidence'],
                    "Emotion": analysis['emotion'],
                    "Topic": analysis['topic'],
                    "Behavior": analysis['behavior'],
                    "Keywords": keyword_str,
                    "Author": comment.get('author', ''),
                    "Likes": comment.get('likes', 0)
                })

            self.data = pd.DataFrame(processed_data)
            self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
            self.data = self.data.sort_values("Datetime").reset_index(drop=True)

            return self.data

        except Exception as e:
            raise gr.Error(f"Failed to retrieve comments: {str(e)}")

    def get_summary_metrics(self):
        """Generate enhanced summary metrics"""
        if self.data is None or self.data.empty:
            return {
                "total": 0, "positive": 0, "neutral": 0, "negative": 0, "mixed": 0,
                "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0, "mixed_pct": 0.0,
                "sentiment_ratio": 0.0, "top_emotion": "N/A", "top_topic": "N/A", "top_behavior": "N/A"
            }

        total_comments = len(self.data)
        sentiment_counts = self.data["Sentiment"].value_counts().to_dict()

        positive = sentiment_counts.get("Positive", 0)
        neutral = sentiment_counts.get("Neutral", 0)
        negative = sentiment_counts.get("Negative", 0)
        mixed = sentiment_counts.get("Mixed", 0)

        def pct(count):
            return round((count / total_comments) * 100, 1) if total_comments else 0.0

        # Get top categories
        top_emotion = self.data["Emotion"].value_counts().index[0] if not self.data["Emotion"].empty else "N/A"
        top_topic = self.data["Topic"].value_counts().index[0] if not self.data["Topic"].empty else "N/A"
        top_behavior = self.data["Behavior"].value_counts().index[0] if not self.data["Behavior"].empty else "N/A"

        return {
            "total": total_comments,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "mixed": mixed,
            "positive_pct": pct(positive),
            "neutral_pct": pct(neutral),
            "negative_pct": pct(negative),
            "mixed_pct": pct(mixed),
            "sentiment_ratio": round(positive / negative, 2) if negative > 0 else float('inf'),
            "top_emotion": top_emotion,
            "top_topic": top_topic,
            "top_behavior": top_behavior
        }

    def export_to_csv(self):
        """Export enhanced data to CSV file"""
        if self.data is None or self.data.empty:
            return None

        try:
            filename = f"multi_level_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_buffer = io.StringIO()
            self.data.to_csv(csv_buffer, index=False)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_buffer.getvalue())

            return filename
        except Exception as e:
            print(f"Error exporting data: {e}")
            return None







class Visualizer:
    """Enhanced Visualizer class with comprehensive plotting capabilities"""

    def __init__(self, data_processor):
        self.processor = data_processor
        self.colors = {
            'Positive': '#28a745',
            'Neutral': '#17a2b8',
            'Negative': '#dc3545',
            'Mixed': '#ffc107'
        }

    def create_sentiment_timeline(self):
        """Create enhanced timeline visualization"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        df_hour = self.processor.data.copy()
        df_hour['Hour'] = df_hour['Datetime'].dt.floor('H')
        grouped = df_hour.groupby(['Hour', 'Sentiment']).size().reset_index(name='Count')

        fig = go.Figure()

        for sentiment in ['Positive', 'Neutral', 'Negative', 'Mixed']:
            data = grouped[grouped['Sentiment'] == sentiment]
            if not data.empty:
                fig.add_trace(go.Scatter(
                    x=data['Hour'],
                    y=data['Count'],
                    mode='markers+lines',
                    name=sentiment,
                    marker=dict(color=self.colors[sentiment], size=8),
                    line=dict(color=self.colors[sentiment], width=2),
                    hovertemplate=f"<b>{sentiment}</b><br>" +
                                  "Time: %{x}<br>" +
                                  "Count: %{y}<br>" +
                                  "<extra></extra>"
                ))

        fig.update_layout(
            title="Sentiment Timeline Analysis",
            xaxis_title="Time",
            yaxis_title="Number of Comments",
            height=400,
            hovermode='x unified',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def create_sentiment_distribution(self):
        """Create enhanced sentiment distribution charts"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        sentiment_counts = self.processor.data['Sentiment'].value_counts()
        colors = [self.colors.get(s, '#6c757d') for s in sentiment_counts.index]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Sentiment Distribution", "Sentiment Counts")
        )

        # Enhanced Pie chart
        fig.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate="<b>%{label}</b><br>" +
                          "Count: %{value}<br>" +
                          "Percentage: %{percent}<br>" +
                          "<extra></extra>"
        ), row=1, col=1)

        # Enhanced Bar chart
        fig.add_trace(go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=colors,
            text=sentiment_counts.values,
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                          "Count: %{y}<br>" +
                          "<extra></extra>"
        ), row=1, col=2)

        fig.update_layout(
            title="Sentiment Distribution Analysis",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        return fig

    def create_keyword_analysis(self):
        """Create enhanced keyword analysis visualization"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        # Extract all keywords and count them
        all_keywords = []
        for keywords_str in self.processor.data['Keywords'].dropna():
            if keywords_str != "N/A":
                keywords = [k.strip() for k in keywords_str.split(',')]
                all_keywords.extend(keywords)

        if not all_keywords:
            fig = go.Figure()
            fig.add_annotation(text="No keywords found", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(15)

        keywords, counts = zip(*top_keywords)

        fig = go.Figure(data=go.Bar(
            x=list(counts),
            y=list(keywords),
            orientation='h',
            marker_color='#17a2b8',
            text=counts,
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>" +
                          "Frequency: %{x}<br>" +
                          "<extra></extra>"
        ))

        fig.update_layout(
            title="Top Keywords Analysis",
            xaxis_title="Frequency",
            yaxis_title="Keywords",
            height=500,
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def create_emotion_analysis(self):
        """Create emotion distribution visualization"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        emotion_counts = self.processor.data['Emotion'].value_counts()

        fig = go.Figure(data=go.Bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a55eea', '#26de81', '#fd79a8'],
            text=emotion_counts.values,
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                          "Count: %{y}<br>" +
                          "<extra></extra>"
        ))

        fig.update_layout(
            title="Emotion Distribution Analysis",
            xaxis_title="Emotions",
            yaxis_title="Count",
            height=400,
            template="plotly_white"
        )

        return fig

    def create_topic_analysis(self):
        """Create topic distribution visualization"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        # Extract all topics (handling comma-separated values)
        all_topics = []
        for topics_str in self.processor.data['Topic'].dropna():
            if topics_str != "General":
                topics = [t.strip() for t in topics_str.split(',')]
                all_topics.extend(topics)
            else:
                all_topics.append("General")

        topic_counts = Counter(all_topics)

        fig = go.Figure(data=go.Pie(
            labels=list(topic_counts.keys()),
            values=list(topic_counts.values()),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate="<b>%{label}</b><br>" +
                          "Count: %{value}<br>" +
                          "Percentage: %{percent}<br>" +
                          "<extra></extra>"
        ))

        fig.update_layout(
            title="Topic Distribution Analysis",
            height=400,
            template="plotly_white"
        )

        return fig

    def create_behavior_analysis(self):
        """Create behavior pattern visualization"""
        if self.processor.data is None or self.processor.data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig

        behavior_counts = self.processor.data['Behavior'].value_counts()

        fig = go.Figure(data=go.Bar(
            x=behavior_counts.values,
            y=behavior_counts.index,
            orientation='h',
            marker_color='#ff7675',
            text=behavior_counts.values,
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>" +
                          "Count: %{x}<br>" +
                          "<extra></extra>"
        ))

        fig.update_layout(
            title="Behavior Pattern Analysis",
            xaxis_title="Count",
            yaxis_title="Behavior Types",
            height=400,
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        if self.processor.data is None or self.processor.data.empty:
            return {
                'timeline': self.create_sentiment_timeline(),
                'distribution': self.create_sentiment_distribution(),
                'keywords': self.create_keyword_analysis(),
                'emotions': self.create_emotion_analysis(),
                'topics': self.create_topic_analysis(),
                'behaviors': self.create_behavior_analysis()
            }

        return {
            'timeline': self.create_sentiment_timeline(),
            'distribution': self.create_sentiment_distribution(),
            'keywords': self.create_keyword_analysis(),
            'emotions': self.create_emotion_analysis(),
            'topics': self.create_topic_analysis(),
            'behaviors': self.create_behavior_analysis()
        }

# Initialize components
data_processor = EnhancedDataProcessor()
visualizer = Visualizer(data_processor)
# Generate sample data
data_processor.generate_sample_data()

def generate_wordcloud(sentiment_filter):
        """Generate word cloud based on sentiment filter"""
        if data_processor.data is None or data_processor.data.empty:
            return None

        try:
            # Filter data based on sentiment
            if sentiment_filter == "All":
                filtered_data = data_processor.data
            else:
                filtered_data = data_processor.data[data_processor.data['Sentiment'] == sentiment_filter]

            if filtered_data.empty:
                return None

            # Combine all text
            all_text = ' '.join(filtered_data['Text'].astype(str))

            # Remove common words and clean text
            cleaned_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
            words = cleaned_text.split()

            # Filter out stop words and short words
            analyzer = MultiLevelSentimentAnalyzer()
            filtered_words = [word for word in words if word not in analyzer.stop_words and len(word) > 3]

            if not filtered_words:
                return None

            word_freq = Counter(filtered_words)

            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(word_freq)

            # Convert to image
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)

            # Save to file
            filename = f"wordcloud_{sentiment_filter.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()

            return filename

        except Exception as e:
            print(f"Error generating word cloud: {e}")
            return None




def analyze_single_comment(comment):
    """Analyze a single comment for quick sentiment analysis"""
    if not comment or comment.strip() == "":
        return "No comment provided", "0.0", "No keywords"

    try:
        analyzer = MultiLevelSentimentAnalyzer()
        analysis = analyzer.multi_level_analysis(comment)
        keywords = analyzer.extract_keywords(comment, 5)

        sentiment = analysis['sentiment']
        confidence = f"{analysis['confidence']:.2f}"
        keyword_str = ", ".join(keywords) if keywords else "No significant keywords found"

        return sentiment, confidence, keyword_str

    except Exception as e:
        return "Error in analysis", "0.0", "Error extracting keywords"

def export_data():
    """Export data and return file path"""
    try:
        filename = data_processor.export_to_csv()
        if filename:
            return gr.File(value=filename, visible=True)
        else:
            return gr.File(visible=False)
    except Exception as e:
        print(f"Export error: {e}")
        return gr.File(visible=False)

# Create Gradio interface
def create_interface():
    # Create Gradio Interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Sentiment Analysis Dashboard") as demo:
        gr.Markdown("""
        # 🎯 Multi-Level Sentiment Analysis Dashboard

        This advanced dashboard provides comprehensive sentiment analysis with emotion detection, topic extraction,
        and behavioral pattern recognition. Analyze YouTube comments or upload your own datasets for deep insights.
        """)

        # Data Input Section
        with gr.Tabs() as input_tabs:
            with gr.Tab("🎬 YouTube Video Analysis"):
                with gr.Row():
                    video_url = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder="https://www.youtube.com/watch?v=..."
                    )
                    url_load_btn = gr.Button("🎬 Analyze Comments", variant="primary")

            with gr.Tab("📁 File Upload Analysis"):
                with gr.Row():
                    file_input = gr.File(
                        label="Upload CSV or Excel File",
                        file_types=[".csv", ".xlsx"]
                    )
                    file_load_btn = gr.Button("📊 Load & Analyze File", variant="primary")

        # Main Dashboard Tabs
        with gr.Tabs():
            # Analytics Dashboard Tab
            with gr.Tab("📊 Analytics Dashboard"):
                # Summary metrics
                metrics = data_processor.get_summary_metrics()

                with gr.Row():
                    total_comments = gr.Number(
                        value=metrics["total"],
                        label="Total Comments",
                        interactive=False
                    )
                    positive_count = gr.Number(
                        value=metrics["positive_pct"],
                        label="Positive %",
                        interactive=False
                    )
                    neutral_count = gr.Number(
                        value=metrics["neutral_pct"],
                        label="Neutral %",
                        interactive=False
                    )
                    negative_count = gr.Number(
                        value=metrics["negative_pct"],
                        label="Negative %",
                        interactive=False
                    )

                with gr.Row():
                    pos_neg_ratio = gr.Number(
                        value=metrics["sentiment_ratio"],
                        label="Positive/Negative Ratio",
                        interactive=False
                    )
                    top_emotion = gr.Textbox(
                        value=metrics["top_emotion"],
                        label="Top Emotion",
                        interactive=False
                    )
                    top_topic = gr.Textbox(
                        value=metrics["top_topic"],
                        label="Top Topic",
                        interactive=False
                    )
                    top_behavior = gr.Textbox(
                        value=metrics["top_behavior"],
                        label="Top Behavior",
                        interactive=False
                    )

                # Visualization
                gr.Markdown("### 📈 Comprehensive Analysis Dashboard")
                
                # Create a grid layout with all visualizations
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Timeline Analysis")
                        timeline_plot = gr.Plot(value=visualizer.create_sentiment_timeline())
                        
                        gr.Markdown("#### Keyword Analysis") 
                        keyword_plot = gr.Plot(value=visualizer.create_keyword_analysis())
                        
                        gr.Markdown("#### Behavior Analysis")
                        behavior_plot = gr.Plot(value=visualizer.create_behavior_analysis())
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Sentiment Distribution")
                        distribution_plot = gr.Plot(value=visualizer.create_sentiment_distribution())
                        
                        gr.Markdown("#### Emotion Analysis")
                        emotion_plot = gr.Plot(value=visualizer.create_emotion_analysis())
                        
                        gr.Markdown("#### Topic Analysis")
                        topic_plot = gr.Plot(value=visualizer.create_topic_analysis())

                # Word Cloud Section
                gr.Markdown("### ☁️ Word Cloud Visualization")

                with gr.Row():
                    sentiment_filter = gr.Dropdown(
                        choices=["All", "Positive", "Neutral", "Negative", "Mixed"],
                        value="All",
                        label="Sentiment Filter"
                    )
                    generate_button = gr.Button("Generate Word Cloud", variant="secondary")

                wordcloud_output = gr.Image(label="Word Cloud")

                # Data Display
                gr.Markdown("### 📋 Detailed Analysis Results")
                comments_display = gr.DataFrame(
                    value=data_processor.data,
                    interactive=False
                )

                with gr.Row():
                    export_btn = gr.Button("💾 Export CSV", variant="secondary")
                    download_component = gr.File(label="Download", visible=False)

            # Quick Analysis Tab
            with gr.Tab("⚡ Quick Sentiment Analyzer"):
                gr.Markdown("""
                ### Quick Multi-Level Analysis Tool
                Analyze sentiment, emotion, topic, and behavior patterns of any text instantly.
                """)

                quick_comment = gr.Textbox(
                    placeholder="Type your comment here...",
                    label="Comment for Analysis",
                    lines=3
                )

                analyze_btn = gr.Button("Analyze Comment", variant="primary")

                with gr.Row():
                    sentiment_result = gr.Textbox(label="Sentiment")
                    confidence_result = gr.Textbox(label="Confidence")
                    keyword_result = gr.Textbox(label="Key Topics")

            # About Tab
            with gr.Row("ℹ️ About This Dashboard"):
                gr.Markdown("""
                ## About This Dashboard

                This advanced dashboard provides comprehensive analysis of text data including sentiment classification,
                emotion detection, topic extraction, and behavioral pattern recognition.

                ### 🚀 Advanced Features:

                - **Multi-Level Analysis**: Sentiment, emotion, topic, and behavior detection
                - **YouTube Integration**: Direct analysis of video comments via YouTube API
                - **File Upload Support**: CSV/Excel file processing
                - **Real-time Visualization**: Interactive charts and graphs
                - **Keyword Extraction**: Automatic identification of important terms
                - **Word Cloud Generation**: Visual representation of text frequency
                - **Export Capabilities**: Download results in CSV format
                - **Quick Analysis Tool**: Instant analysis of individual comments

                ### 📊 Analysis Dimensions:

                1. **Sentiment**: Positive, Negative, Neutral, Mixed
                2. **Emotion**: Joy, Anger, Sadness, Fear, Surprise, etc.
                3. **Topic**: Product, Service, Delivery, Pricing, etc.
                4. **Behavior**: Promoting, Complaining, Supporting, etc.

                ### 📝 How to Use:

                1. **Data Input**: Upload a file or enter a YouTube URL
                2. **Dashboard Review**: Examine metrics and visualizations
                3. **Deep Dive**: Explore individual analysis tabs
                4. **Quick Test**: Use the quick analyzer for single comments
                5. **Export**: Download results for further analysis

                ### 📋 File Requirements:

                - Format: CSV or Excel (.csv, .xlsx)
                - Required column: 'Text' (containing comments/reviews)
                - Optional: 'Datetime' (will be auto-generated if missing)

                ---

                **Developed by [Anaclet UKURIKIYEYEZU](https://portofolio-pi-lac.vercel.app/)**

                ### 📞 Contact Information:
                - **WhatsApp**: [+250 786 698 014](https://wa.me/250786698014)
                - **Email**: [anaclet.ukurikiyeyezu@aims.ac.rw](mailto:anaclet.ukurikiyeyezu@aims.ac.rw)
                """)

        # Event Handlers - FIXED: Added all required outputs

        # File upload event
        file_load_btn.click(
            fn=lambda file: load_and_update_all_components(file, None),
            inputs=[file_input],
            outputs=[
                total_comments, positive_count, neutral_count,
                negative_count, pos_neg_ratio, top_emotion,
                top_topic, top_behavior, timeline_plot, distribution_plot,
                keyword_plot, emotion_plot, topic_plot, behavior_plot,
                comments_display
            ]
        )

        # YouTube analysis event
        url_load_btn.click(
            fn=lambda url: load_and_update_all_components(None, url),
            inputs=[video_url],
            outputs=[
                total_comments, positive_count, neutral_count,
                negative_count, pos_neg_ratio, top_emotion,
                top_topic, top_behavior, timeline_plot, distribution_plot,
                keyword_plot, emotion_plot, topic_plot, behavior_plot,
                comments_display
            ]
        )

        # Word cloud generation event
        generate_button.click(
            fn=generate_wordcloud,
            inputs=[sentiment_filter],
            outputs=[wordcloud_output]
        )

        # Comment analysis event
        analyze_btn.click(
            fn=analyze_single_comment,
            inputs=[quick_comment],
            outputs=[sentiment_result, confidence_result, keyword_result]
        )

        # Export to CSV event
        export_btn.click(
            fn=export_data,
            inputs=[],
            outputs=[download_component]
        )

    return demo

# Updated load_and_update_all_components to match outputs
def load_and_update_all_components(file, video_url):
    """Load data and update all dashboard components"""
    try:
        if video_url and video_url.strip():
            # Process YouTube comments
            data_processor.process_youtube_comments(video_url.strip())
        elif file:
            # Process uploaded file
            data_processor.process_file(file)
        else:
            # Use sample data
            data_processor.generate_sample_data()

        # Get updated metrics
        metrics = data_processor.get_summary_metrics()

        # Create updated visualizations
        timeline_fig = visualizer.create_sentiment_timeline()
        distribution_fig = visualizer.create_sentiment_distribution()
        keyword_fig = visualizer.create_keyword_analysis()
        emotion_fig = visualizer.create_emotion_analysis()
        topic_fig = visualizer.create_topic_analysis()
        behavior_fig = visualizer.create_behavior_analysis()


        return (
            metrics["total"],
            metrics["positive_pct"],
            metrics["neutral_pct"],
            metrics["negative_pct"],
            metrics["sentiment_ratio"],
            metrics["top_emotion"],
            metrics["top_topic"],
            metrics["top_behavior"],
            timeline_fig,
            distribution_fig,
            keyword_fig,
            emotion_fig,
            topic_fig,
            behavior_fig,
            data_processor.data
        )

    except Exception as e:
        # Return error message and current state
        print(f"Error in load_and_update_all_components: {e}")
        metrics = data_processor.get_summary_metrics()
        return (
            metrics["total"],
            metrics["positive_pct"],
            metrics["neutral_pct"],
            metrics["negative_pct"],
            metrics["sentiment_ratio"],
            metrics["top_emotion"],
            metrics["top_topic"],
            metrics["top_behavior"],
            visualizer.create_sentiment_timeline(),
            visualizer.create_sentiment_distribution(),
            visualizer.create_keyword_analysis(),
            visualizer.create_emotion_analysis(),
            visualizer.create_topic_analysis(),
            visualizer.create_behavior_analysis(),
            data_processor.data
        )


# Main execution
if __name__ == "__main__":
    # Get port from environment
    port = int(os.environ.get("PORT", 7860))
    
    # Create and launch app
    demo = create_interface()
    
    print(f"Starting server on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=False,
        show_error=True
    )
































