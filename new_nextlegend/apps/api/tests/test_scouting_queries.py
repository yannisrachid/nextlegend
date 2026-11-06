import unittest
from scouting_queries import compile_query, PlayerQuery

class QuerySafetyTests(unittest.TestCase):
    def test_model_report_link_normalization(self):
        from scouting_intelligence import normalize_report_links
        self.assertEqual(normalize_report_links('[Youth](#/youth-scouting/reports?youth_id=3)'), '[Youth](/youth-scouting/reports?youth_id=3)')
        self.assertEqual(normalize_report_links('[Pro](sandbox:/report?player_id=2)'), '[Pro](/report?player_id=2)')
        self.assertEqual(normalize_report_links('[Other](sandbox:/private)'), '[Other](sandbox:/private)')

    def test_injection_is_bound(self):
        attack="'; DROP TABLE players; --"
        sql,p=compile_query({'club':attack,'competitions':[attack]})
        self.assertNotIn(attack,sql)
        self.assertEqual(p['competitions0'],attack)
        self.assertIn(attack,p['club'])

    def test_arbitrary_identifiers_rejected(self):
        for key in ['password','score; DELETE FROM players','pg_sleep(30)']:
            with self.assertRaises(ValueError): compile_query({'order_by':key})

    def test_constraints_not_silently_dropped(self):
        with self.assertRaises(ValueError): compile_query({'source':'eyeball','filters':[{'metric':'market_value_eur','value':100}]})
        with self.assertRaises(ValueError): compile_query({'filters':[{'metric':'unknown_stat','value':10}]})
        with self.assertRaises(ValueError): PlayerQuery(min_age=25,max_age=20)
        with self.assertRaises(ValueError): PlayerQuery(limit=10000)
        with self.assertRaises(ValueError): PlayerQuery(season='all')

    def test_unknown_not_zero(self):
        sql,_=compile_query({'filters':[{'metric':'fm_pa','value':140},{'metric':'market_value_eur','operator':'lte','value':20_000_000}]})
        self.assertNotIn('COALESCE',sql)
        self.assertIn("m.status='accepted'",sql)
        self.assertIn("m.provider='wyscout'",sql)
        self.assertIn('m.provider_id=p.id::text',sql)

    def test_cohort_denominators(self):
        sql,_=compile_query({'group_by':'competition','metrics':['xg_per_90'],'min_group_size':20})
        self.assertIn('COUNT(DISTINCT player_id)',sql)
        self.assertIn('COUNT("xg_per_90")',sql)
        self.assertIn('HAVING COUNT(*)>=:min_group_size',sql)

    def test_youth_numeric_cast_guarded(self):
        sql,p=compile_query({'source':'eyeball','metrics':['goals']},['goals',"evil');DROP TABLE players;--"])
        self.assertIn('CASE WHEN',sql)
        self.assertNotIn('DROP TABLE',sql)
        self.assertEqual(p['key_0'],'goals')

if __name__=='__main__': unittest.main()
