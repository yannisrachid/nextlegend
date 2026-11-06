import unittest
from unittest.mock import Mock
from fm26 import match_players
from scouting_intelligence import search_players, Identity, Search, safe_url

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.fm=[dict(uid=10,unique_id=999,name='Jules Martin',full_name='Jules Martin',birth_date='2005-03-02',club_name='Lens')]
        self.target=dict(provider='wyscout',provider_id='999',name='Jules Martin',birth_date='2005-03-02',club_name='Lens')

    def test_exact_identity(self):
        self.assertEqual(match_players(self.fm,[self.target])[0]['status'],'accepted')

    def test_equal_numeric_id_does_not_establish_match(self):
        self.assertEqual(match_players(self.fm,[dict(self.target,name='Someone Else')]),[])

    def test_wrong_dob_rejected(self):
        self.assertEqual(match_players(self.fm,[dict(self.target,birth_date='2004-03-02')]),[])

    def test_unique_full_name_year_and_club(self):
        self.assertEqual(match_players(self.fm,[dict(self.target,birth_date=None,birth_year=2005)])[0]['status'],'accepted')

    def test_estimated_age_never_auto_accepted(self):
        t=dict(self.target,birth_date=None,source_age=21,observed_at='2026-09-22')
        self.assertEqual(match_players(self.fm,[t])[0]['status'],'review')

    def test_ambiguous_fm_and_target(self):
        duplicated=self.fm+[dict(self.fm[0],uid=11)]
        self.assertTrue(all(m['status']=='review' for m in match_players(duplicated,[self.target])))
        self.assertTrue(all(m['status']=='review' for m in match_players(self.fm,[self.target,dict(self.target,provider_id='333')])))

    def test_tm_alias_needs_corroboration(self):
        t=dict(self.target,name='J. Martin',birth_date=None,alias='Jules Martin',alternate_birth_date='2005-03-02')
        self.assertEqual(match_players(self.fm,[t])[0]['status'],'accepted')
        self.assertEqual(match_players(self.fm,[dict(t,name='L. Martinez')]),[])

class ToolTests(unittest.TestCase):
    def test_fm_cannot_satisfy_observed_minutes(self):
        session=Mock()
        self.assertIn('error',search_players(session,{'source':'fm','min_minutes':450}))
        session.execute.assert_not_called()

    def test_provider_and_limits(self):
        for kwargs in ({'source':'fm','player_id':8252},{'source':'auth_users','player_id':1}):
            with self.assertRaises(ValueError): Identity(**kwargs)
        with self.assertRaises(ValueError): Search(source='wyscout',limit=10000)

    def test_untrusted_search_is_bound_not_sql(self):
        session=Mock(); session.execute.return_value.mappings.return_value=[]
        injection="'; DROP TABLE players; --"
        search_players(session,{'source':'wyscout','query':injection})
        sql,parameters=session.execute.call_args.args
        self.assertNotIn(injection,str(sql)); self.assertIn(injection,parameters['query'])

    def test_citation_schemes(self):
        self.assertTrue(safe_url('https://example.com/news'))
        self.assertFalse(safe_url('javascript:alert(1)'))
        self.assertFalse(safe_url('file:///etc/passwd'))

if __name__=='__main__': unittest.main()
