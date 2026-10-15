from django.db import migrations,models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("engine","0108_profile_registration_survey_fields")]
    operations = [
        migrations.CreateModel(name="AutoLabelStreamState",fields=[
            ("job",models.OneToOneField(primary_key=True,serialize=False,on_delete=django.db.models.deletion.CASCADE,to="engine.job")),
            ("state",models.JSONField(default=dict))]),
        migrations.CreateModel(name="AutoLabelStreamOperation",fields=[
            ("id",models.BigAutoField(primary_key=True,serialize=False)),
            ("job",models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,to="engine.job")),
            ("operation_id",models.CharField(max_length=200)),("digest",models.CharField(max_length=64)),
            ("result",models.JSONField(default=dict))],
            options={"unique_together":{("job","operation_id")}}),
    ]
